import fitz  # PyMuPDF
from pdf2image import convert_from_path
from PIL import Image, ImageDraw

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
