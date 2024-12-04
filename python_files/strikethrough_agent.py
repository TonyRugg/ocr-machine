import cv2
import os
import time
import numpy as np
import pdf2image
from trp import Document
from tqdm import tqdm

def has_strikethrough(pdf_path, geometry, debug=False):
    """
    Detects whether a word region in a PDF contains a strikethrough line.
    
    This function analyzes a specific word region within a PDF page to determine if it 
    contains a strikethrough line. It uses computer vision techniques to identify 
    horizontal or vertical lines that cross through text, taking into account the 
    text's orientation as provided by the Textract Response Parser.
    
    Args:
        pdf_path (str): Path to the PDF file containing the word
        geometry (dict): Dictionary containing geometric information about the word:
            - bbox (dict): Bounding box with 'Left', 'Top', 'Width', 'Height' (normalized coordinates)
            - page_num (int): Page number (0-based)
            - orientation (float): Text orientation in degrees from Textract
        debug (bool, optional): If True, saves debug images to disk. Defaults to False.
    
    Returns:
        bool: True if a strikethrough line is detected, False otherwise
    """
    debug_dir = "debug_images"
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        timestamp = int(time.time() * 1000)

    # Extract geometry information
    bbox = geometry['bbox']
    page_num = geometry['page_num']
    orientation = geometry['orientation']

    # Load and convert PDF page to image
    pages = pdf2image.convert_from_path(pdf_path, dpi=400)
    image = np.array(pages[page_num])
    
    # Convert normalized coordinates to pixels
    img_height, img_width = image.shape[:2]
    left_px = int(bbox['Left'] * img_width)
    top_px = int(bbox['Top'] * img_height)
    width_px = int(bbox['Width'] * img_width)
    height_px = int(bbox['Height'] * img_height)
    
    if debug:
        print(f"Processing region: {width_px}x{height_px} pixels at ({left_px}, {top_px})")
        print(f"Page orientation: {orientation} degrees")

    # Extract word region from image
    word_region = image[top_px:top_px + height_px, left_px:left_px + width_px]

    # Image preprocessing steps
    # Convert to grayscale and apply Gaussian blur to reduce noise
    gray = cv2.cvtColor(word_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Convert to binary image using Otsu's thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"binary_{timestamp}.png"), binary)

    # Determine morphological operation direction based on orientation
    is_vertical = abs(orientation) > 45

    # Create kernels for morphological operations
    # Close small gaps in text while preserving potential strikethrough lines
    if is_vertical:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height_px // 4, 3)))
    else:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width_px // 4, 3), 1))

    # Apply morphological closing to connect nearby components
    closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"closed_binary_{timestamp}.png"), closed_binary)

    # Detect potential strikethrough lines using morphological opening
    detected_lines = cv2.morphologyEx(closed_binary, cv2.MORPH_OPEN, line_kernel, iterations=1)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"detected_lines_{timestamp}.png"), detected_lines)

    # Find contours in the detected lines
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Analyze each detected contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Apply geometric criteria based on orientation
        if is_vertical:
            # For vertical text, check for vertical strikethrough
            if h < height_px * 0.45:  # Line must span at least 45% of height
                continue
            if w >= width_px * 0.25:  # Line shouldn't be too thick
                continue
            
            # Check horizontal position (should be roughly centered)
            center_x = x + w/2
            if not (width_px * 0.1 <= center_x <= width_px * 0.9):
                continue
                
        else:
            # For horizontal text, check for horizontal strikethrough
            if h >= height_px * 0.25:  # Line shouldn't be too thick
                continue
            if w <= width_px * 0.4:  # Line must span at least 40% of width
                continue
            
            # Check vertical position (should be roughly centered)
            center_y = y + h/2
            if not (height_px * 0.3 <= center_y <= height_px * 0.7):
                continue

        # Check line density to ensure it's a solid line
        line_region = closed_binary[y:y+h, x:x+w]
        density = np.sum(line_region) / (255 * w * h)
        if density < 0.35:  # Line should be reasonably solid
            continue

        if debug:
            debug_image = word_region.copy()
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"found_line_{timestamp}.png"), debug_image)
            print(f"Strikethrough detected: {w}x{h} pixels at ({x}, {y})")
            print(f"Line density: {density:.2f}")

        return True

    if debug:
        print("No strikethrough detected")
        
    return False


def clean_response_with_strikethrough(response, pdf_path, debug=False):
    """
    Cleans a Textract response by removing words with strikethrough.
    Uses textract-response-parser for easier document handling.
    
    Args:
    response (dict): The Textract response dictionary
    pdf_path (str): Path to the PDF file for strikethrough detection
    debug (bool): If True, debug images will be saved by `has_strikethrough`

    Returns:
    tuple: (cleaned response dict, removed bounding boxes list)
    """
    # Convert response to Document object for easier handling
    doc = Document(response)
    removed = []  # List to store removed bounding boxes
    removed_ids = set()  # Track removed block IDs
    
    # Create new blocks list
    new_blocks = []
    
    for block in tqdm(response.get("Blocks", [])):
        if block["BlockType"] == "WORD":
            # Get page object to access orientation
            page_num = block.get("Page", 1)
            page = doc.pages[page_num - 1]
            orientation = page.custom.get('PageOrientationBasedOnWords', 0)
            
            # Package geometry info into a dict
            geometry = {
                'bbox': block["Geometry"]["BoundingBox"],
                'page_num': page_num - 1,  # Convert to 0-based indexing
                'orientation': orientation
            }
            
            has_strike = has_strikethrough( 
                pdf_path=pdf_path,
                geometry=geometry,
                debug=debug
            )
            
            if has_strike:
                removed.append(block["Geometry"]["BoundingBox"])
                removed_ids.add(block["Id"])
                continue
                
        new_blocks.append(block)
    
    # Update relationships in remaining blocks
    for block in new_blocks:
        if "Relationships" in block:
            updated_relationships = []
            for relationship in block["Relationships"]:
                if relationship["Type"] == "CHILD":
                    # Filter out removed IDs from CHILD relationships
                    relationship["Ids"] = [
                        child_id for child_id in relationship["Ids"] 
                        if child_id not in removed_ids
                    ]
                updated_relationships.append(relationship)
            block["Relationships"] = updated_relationships

    # Update response with cleaned blocks
    response["Blocks"] = new_blocks
    
    return response, removed


def strikethrough_manager():
    '''
    IS this top-level needed above the has_strikethrough?
    Checks the page for strikethroughs and 
    Returns a list of bounding boxes with true strikes to be removed
    ?Should this also return a cleaned json with no strikes?
    '''

