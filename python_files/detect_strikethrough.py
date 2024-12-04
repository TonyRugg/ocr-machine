import cv2
import os
import time
import numpy as np
import pdf2image

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









__doc__ = """
Strikethrough Detection in PDF Word Regions

This script defines a function, `has_strikethrough`, which detects whether a word region in a PDF contains a strikethrough line.
It leverages computer vision techniques to analyze horizontal or vertical lines within specified bounding boxes on a page.

Dependencies:
    - OpenCV (cv2): For image processing and contour analysis.
    - pdf2image: For converting PDF pages into images.
    - NumPy: For numerical computations.
    - time: For timestamp-based debugging.
    - os: For directory management.

Function:
    has_strikethrough(pdf_path, geometry, debug=False):
        Determines if a word region in a PDF page contains a strikethrough line.

        Arguments:
            - pdf_path (str): Path to the PDF file.
            - geometry (dict): Dictionary containing the word's geometric information:
                - 'bbox' (dict): Normalized bounding box with 'Left', 'Top', 'Width', 'Height'.
                - 'page_num' (int): Zero-based page index.
                - 'orientation' (float): Text orientation in degrees.
            - debug (bool, optional): If True, saves intermediate debug images to the `debug_images` directory.

        Returns:
            - bool: True if a strikethrough line is detected, False otherwise.

Implementation Details:
    - The function extracts the specified word region from the PDF page using `pdf2image`.
    - The word region undergoes preprocessing:
        - Grayscale conversion and Gaussian blurring for noise reduction.
        - Binary thresholding using Otsu's method to highlight text and lines.
    - Morphological operations:
        - Closing is applied to connect broken text components.
        - Opening with directional kernels is used to isolate potential strikethrough lines.
    - Contour Analysis:
        - Contours of potential lines are extracted and filtered based on geometric criteria:
            - Orientation-specific thresholds for thickness, length, and position.
            - Line density ensures detected lines are reasonably solid.

Debug Mode:
    - If `debug=True`, the function saves intermediate steps as images in the `debug_images` directory:
        - Binary image (`binary_TIMESTAMP.png`)
        - Post-closing image (`closed_binary_TIMESTAMP.png`)
        - Detected lines image (`detected_lines_TIMESTAMP.png`)
        - Debug output showing detected lines with bounding boxes.

Usage:
    - Import the `has_strikethrough` function:
        ```python
        from script_name import has_strikethrough
        ```
    - Call the function with the required arguments:
        ```python
        geometry = {
            'bbox': {'Left': 0.1, 'Top': 0.2, 'Width': 0.4, 'Height': 0.1},
            'page_num': 0,
            'orientation': 0
        }
        result = has_strikethrough("path/to/pdf.pdf", geometry, debug=True)
        print("Strikethrough detected:", result)
        ```

Output:
    - Boolean indicating the presence of a strikethrough line.
    - Debug images saved to the `debug_images` directory when `debug=True`.

Notes:
    - Assumes normalized coordinates for bounding boxes (as percentages of page width/height).
    - Handles both horizontal and vertical text orientations.
    - Requires the `pdf2image` library to be correctly configured with poppler utilities.

"""
