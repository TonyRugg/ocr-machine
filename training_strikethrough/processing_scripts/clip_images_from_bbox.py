import json
import fitz
from PIL import Image
import os
from tqdm import tqdm

def extract_word_images(pdf_path, json_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Load PDF
    pdf_document = fitz.open(pdf_path)
    
    # Load JSON
    with open(json_path, 'r') as f:
        textract_data = json.load(f)
    
    # Extract words and bounding boxes
    for block in tqdm(textract_data["Blocks"]):
        if block["BlockType"] == "WORD":
            page_index = block["Page"] - 1  # Pages in PyMuPDF are 0-indexed
            page = pdf_document[page_index]
            
            # Use mediabox_size to get page dimensions
            page_width, page_height = page.mediabox_size
            
            bbox = block["Geometry"]["BoundingBox"]
            
            # Convert normalized bounding box to absolute coordinates
            x0 = int(bbox["Left"] * page_width)
            y0 = int(bbox["Top"] * page_height)
            x1 = int((bbox["Left"] + bbox["Width"]) * page_width)
            y1 = int((bbox["Top"] + bbox["Height"]) * page_height)
            
            # Crop the image
            pix = page.get_pixmap(clip=fitz.Rect(x0, y0, x1, y1))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Save the image
            output_path = f"{output_folder}/{block['Id']}.jpeg"
            img.save(output_path, "JPEG")

    pdf_document.close()

if __name__ == "__main__":
    FNAME = 'scanned_output_20241202_142009'
    source_pdf = f"trainingSet/false_pdf/{FNAME}.pdf"
    source_json = f"trainingSet/false_json/{FNAME}.json"
    destination_folder = f"trainingSet/false_images/{FNAME}"
    extract_word_images(source_pdf, source_json, destination_folder)
