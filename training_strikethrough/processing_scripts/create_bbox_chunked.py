import json
import os
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars

def process_word(block, page_image, page_width, page_height, output_folder):
    if block["BlockType"] != "WORD":
        return

    # Extract bounding box
    bbox = block["Geometry"]["BoundingBox"]
    x0 = int(bbox["Left"] * page_width)
    y0 = int(bbox["Top"] * page_height)
    x1 = int((bbox["Left"] + bbox["Width"]) * page_width)
    y1 = int((bbox["Top"] + bbox["Height"]) * page_height)

    # Crop the word image
    cropped_image = page_image.crop((x0, y0, x1, y1))
    if cropped_image.width == 0 or cropped_image.height == 0:
        print(f"Empty cropped image for block {block['Id']}, skipping.")
        return

    # Save the cropped word image
    output_path = f"{output_folder}/{block['Id']}.jpeg"
    cropped_image.save(output_path, "JPEG")

def extract_word_images(pdf_path, json_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON
    with open(json_path, 'r') as f:
        textract_data = json.load(f)

    # Group WORD blocks by page
    word_blocks_by_page = {}
    for block in textract_data["Blocks"]:
        if block["BlockType"] == "WORD":
            page_number = block["Page"]
            if page_number not in word_blocks_by_page:
                word_blocks_by_page[page_number] = []
            word_blocks_by_page[page_number].append(block)

    # Initialize the progress bar for the entire document
    total_pages = len(word_blocks_by_page)
    with tqdm(total=total_pages, desc="Processing Document", unit="page") as doc_bar:
        # Process the PDF page by page
        for page_number, word_blocks in word_blocks_by_page.items():
            # Convert the specific page to an image
            page_images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=150)
            if not page_images:
                print(f"No image generated for page {page_number}, skipping.")
                doc_bar.update(1)
                continue

            page_image = page_images[0]  # Only one image since it's one page
            page_width, page_height = page_image.size

            # Create a folder for each page (optional)
            page_output_folder = os.path.join(output_folder, f"page_{page_number}")
            os.makedirs(page_output_folder, exist_ok=True)

            # Process all WORD blocks for this page
            for block in word_blocks:
                process_word(block, page_image, page_width, page_height, page_output_folder)

            # Update the progress bar after processing the page
            doc_bar.update(1)

    print("Processing complete.")

if __name__ == "__main__":
    FNAME = 'STRIKEdallas-tx-3_chunk_1'
    source_pdf = f"trainingSet/true_pdf/{FNAME}.pdf"
    source_json = f"trainingSet/true_json/{FNAME}.json"
    destination_folder = f"trainingSet/true_images/{FNAME}"
    extract_word_images(source_pdf, source_json, destination_folder)
