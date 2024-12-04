import cv2
import os
import time
import numpy as np
import pdf2image
import pytesseract
from PIL import Image

def analyze_pixel_distribution(image, debug=False):
    """
    Analyze pixel distribution to determine text orientation
    Returns: True if vertical, False if horizontal
    """
    # Convert to binary if not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        binary = image
        
    # Get height and width
    height, width = binary.shape
    
    # Calculate vertical and horizontal pixel sums
    vertical_projection = np.sum(binary, axis=0)  # Sum each column
    horizontal_projection = np.sum(binary, axis=1)  # Sum each row
    
    # Calculate spread measures
    v_std = np.std(vertical_projection)
    h_std = np.std(horizontal_projection)
    
    if debug:
        print(f"Vertical std: {v_std}, Horizontal std: {h_std}")
        print(f"Binary shape: {binary.shape}")
    
    # If vertical text, horizontal spread (std) will be larger
    # because pixels are concentrated in fewer rows
    return v_std < h_std  # or simply return h_std > v_std

def has_strikethrough(pdf_path, left, top, width, height, page_num=0, word_id=0, debug=False):
    """
    Enhanced strikethrough detection with full line requirement for small boxes
    """
    debug_dir = "debug_images"
    if debug:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"\nAnalyzing word_id: {word_id}")

    # Load and convert PDF page
    pages = pdf2image.convert_from_path(pdf_path, dpi=300)
    image = np.array(pages[page_num])

    # Convert coordinates to pixels
    img_height, img_width = image.shape[:2]
    left_px = int(left * img_width)
    top_px = int(top * img_height)
    width_px = int(width * img_width)
    height_px = int(height * img_height)
    
    if debug:
        print(f"Original box dimensions: {width_px}x{height_px} pixels")
        print(f"Position: left={left_px}, top={top_px}")

    # Check if this is a small box
    SMALL_BOX_THRESHOLD = 40  # pixels
    is_small_box = max(width_px, height_px) < SMALL_BOX_THRESHOLD
    
    if debug:
        print(f"Small box check: {'YES' if is_small_box else 'NO'} (threshold: {SMALL_BOX_THRESHOLD}px)")

    # Extract word region
    word_region = image[top_px:top_px + height_px, left_px:left_px + width_px]

    # Preprocessing
    gray = cv2.cvtColor(word_region, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if debug:
        timestamp = int(time.time() * 1000)
        cv2.imwrite(os.path.join(debug_dir, f"word_{word_id}_binary_{timestamp}.png"), binary)

    # Determine orientation
    if is_small_box:
        if debug:
            print("Using pixel distribution analysis for orientation")
        is_vertical = analyze_pixel_distribution(binary, debug)
    else:
        is_vertical = height_px > width_px

    if debug:
        print(f"Orientation detection: {'VERTICAL' if is_vertical else 'HORIZONTAL'}")

    # Morphological operations
    if is_vertical:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height_px // 4, 3)))
    else:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width_px // 4, 3), 1))

    closed_binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"word_{word_id}_closed_binary_{timestamp}.png"), closed_binary)

    # Detect lines
    detected_lines = cv2.morphologyEx(closed_binary, cv2.MORPH_OPEN, line_kernel, iterations=1)

    if debug:
        cv2.imwrite(os.path.join(debug_dir, f"word_{word_id}_lines_{timestamp}.png"), detected_lines)

    # Find contours
    contours, _ = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    strikethrough_detected = False
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if is_small_box:
            if is_vertical:
                # For vertical text in small box, require full height line
                if h < height_px:
                    if debug:
                        print(f"Failed small box full height requirement: {h} < {height_px}")
                    continue
                # Check horizontal position
                center_x = x + w/2
                if not (width_px * 0.1 <= center_x <= width_px * 0.9):
                    if debug:
                        print(f"Failed position requirement: {center_x} not in range ({width_px * 0.1}, {width_px * 0.9})")
                    continue
            else:
                # For horizontal text in small box, require full width line
                if w < width_px:
                    if debug:
                        print(f"Failed small box full width requirement: {w} < {width_px}")
                    continue
                # Check vertical position
                center_y = y + h/2
                if not (height_px * 0.3 <= center_y <= height_px * 0.7):
                    if debug:
                        print(f"Failed position requirement: {center_y} not in range ({height_px * 0.3}, {height_px * 0.7})")
                    continue
        else:
            if is_vertical:
                # Regular checks for normal sized vertical text
                if h < height_px * 0.45:
                    if debug:
                        print(f"Failed height requirement: {h} < {height_px * 0.45}")
                    continue
                if w >= width_px * 0.25:
                    if debug:
                        print(f"Failed width requirement: {w} >= {width_px * 0.25}")
                    continue
                center_x = x + w/2
                if not (width_px * 0.1 <= center_x <= width_px * 0.9):
                    if debug:
                        print(f"Failed position requirement: {center_x} not in range ({width_px * 0.1}, {width_px * 0.9})")
                    continue
                top_margin = y
                bottom_margin = height_px - (y + h)
                if top_margin > height_px * 0.3 or bottom_margin > height_px * 0.3:
                    if debug:
                        print(f"Failed margin requirements: top={top_margin}, bottom={bottom_margin}")
                    continue
            else:
                # Regular checks for normal sized horizontal text
                if h >= height_px * 0.25:
                    if debug:
                        print(f"Failed height requirement: {h} >= {height_px * 0.25}")
                    continue
                if w <= width_px * 0.4:
                    if debug:
                        print(f"Failed width requirement: {w} <= {width_px * 0.4}")
                    continue
                center_y = y + h/2
                if not (height_px * 0.3 <= center_y <= height_px * 0.7):
                    if debug:
                        print(f"Failed position requirement: {center_y} not in range ({height_px * 0.3}, {height_px * 0.7})")
                    continue

        # Density check
        line_region = closed_binary[y:y+h, x:x+w]
        density = np.sum(line_region) / (255 * w * h)
        if density < 0.35:
            if debug:
                print(f"Failed density requirement: {density} < 0.35")
            continue

        strikethrough_detected = True
        if debug:
            print("Contour passed all checks:")
            print(f"  Position: ({x}, {y})")
            print(f"  Size: {w}x{h}")
            print(f"  Density: {density:.2f}")

        if debug:
            debug_image = word_region.copy()
            cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"word_{word_id}_line_{timestamp}_{x}_{y}.png"), debug_image)

    if debug and not strikethrough_detected:
        cv2.imwrite(os.path.join(debug_dir, f"word_{word_id}_no_strikethrough_{timestamp}.png"), binary)
        print("No strikethrough detected")

    return strikethrough_detected