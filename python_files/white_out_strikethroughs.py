from pdf2image import convert_from_path
from PIL import ImageDraw

def clean_pdf_with_removed_boxes(pdf_path, removed, output_pdf_path):
    """
    Cleans a PDF by filling bounding boxes from the `removed` variable with white.
    
    Args:
    pdf_path (str): Path to the original PDF file.
    removed (list): List of bounding boxes to be filled with white.
                    Each bounding box should be a dict with `Left`, `Top`, `Width`, and `Height`.
    output_pdf_path (str): Path to save the cleaned PDF file.
    """
    # Convert PDF to images (one per page)
    pages = convert_from_path(pdf_path, dpi=300)
    img_width, img_height = pages[0].size  # Assume uniform page dimensions

    # Prepare a list to store modified images
    modified_images = []

    for page_num, page_image in enumerate(pages):
        # Create a copy of the page as an editable PIL Image
        editable_page = page_image.copy()
        draw = ImageDraw.Draw(editable_page)

        # Fill bounding boxes for the current page
        for bbox in removed:
            left = int(bbox["Left"] * img_width)
            top = int(bbox["Top"] * img_height)
            width = int(bbox["Width"] * img_width)
            height = int(bbox["Height"] * img_height)

            # Fill the bounding box with white
            draw.rectangle([left, top, left + width, top + height], fill="white")

        # Add modified page to the list
        modified_images.append(editable_page)

    # Save all modified pages as a cleaned PDF
    modified_images[0].save(
        output_pdf_path,
        save_all=True,
        append_images=modified_images[1:],
        resolution=300
    )
    print(f"Cleaned PDF saved to: {output_pdf_path}")












__doc__ = """
Clean PDF by Removing Content with Bounding Boxes

This script provides functionality to clean a PDF by filling specific bounding box regions with white. 
The bounding boxes are defined as a list of dictionaries, each specifying the region to be removed 
on a page of the PDF. The output is a modified PDF with the specified regions obscured.

Dependencies:
    - pdf2image: For converting PDF pages to images.
    - Pillow (PIL): For modifying images and saving the cleaned PDF.

Function:
    clean_pdf_with_removed_boxes(pdf_path, removed, output_pdf_path):
        Processes a PDF file, fills specified bounding boxes with white, and saves the cleaned PDF.

Arguments:
    - `pdf_path` (str): Path to the original PDF file.
    - `removed` (list): List of bounding boxes to be removed. Each bounding box should be a dictionary with:
        - `Left` (float): Left coordinate of the box as a fraction of the page width.
        - `Top` (float): Top coordinate of the box as a fraction of the page height.
        - `Width` (float): Width of the box as a fraction of the page width.
        - `Height` (float): Height of the box as a fraction of the page height.
    - `output_pdf_path` (str): Path to save the cleaned PDF file.

Implementation:
    - Converts the input PDF to images (one image per page) using `pdf2image`.
    - Iterates through each page and fills the specified bounding boxes with white.
    - Saves the modified images as a new PDF.

Usage:
    - Import the function:
        ```python
        from script_name import clean_pdf_with_removed_boxes
        ```
    - Call the function with the necessary arguments:
        ```python
        removed_boxes = [
            {"Left": 0.1, "Top": 0.2, "Width": 0.3, "Height": 0.1},  # Example bounding box
            {"Left": 0.4, "Top": 0.5, "Width": 0.2, "Height": 0.2}
        ]
        clean_pdf_with_removed_boxes("input.pdf", removed_boxes, "cleaned_output.pdf")
        ```

Notes:
    - Bounding box coordinates should be normalized (fractions of the page dimensions).
    - Assumes all pages in the PDF have uniform dimensions.
    - The script uses a resolution of 300 DPI for the conversion and output.

Output:
    - A new PDF file with the specified bounding boxes filled with white.

Example:
    - Input PDF with redacted regions:
        ```python
        clean_pdf_with_removed_boxes("original.pdf", removed_boxes, "redacted.pdf")
        ```
    - The cleaned PDF is saved to the specified output path.

"""
