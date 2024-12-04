from detect_strikethrough import has_strikethrough
from tqdm import tqdm

def clean_response_with_strikethrough(response, pdf_path, debug=False):
    """
    Cleans a Textract response dictionary by removing "WORD" BlockTypes with a strikethrough and updates related references.
    
    Args:
    response (dict): The Textract response dictionary.
    pdf_path (str): Path to the PDF file for strikethrough detection.
    debug (bool): If True, debug images will be saved by `has_strikethrough`.

    Returns:
    tuple: (cleaned response, removed bounding boxes list)
    """
    removed = []  # List to store removed bounding boxes
    removed_ids = set()  # Track removed block IDs

    # Filter out "WORD" blocks with strikethrough
    new_blocks = []
    for block in tqdm(response.get("Blocks", [])):
        if block["BlockType"] == "WORD":
            bbox = block["Geometry"]["BoundingBox"]
            page_num = block.get("Page", 1) - 1  # Textract is 1-based, but PDF pages are 0-based
            has_strike = has_strikethrough(
                pdf_path, bbox["Left"], bbox["Top"], bbox["Width"], bbox["Height"], page_num, debug
            )
            if has_strike:
                removed.append(bbox)
                removed_ids.add(block["Id"])
                continue  # Skip adding this block
        new_blocks.append(block)

    # Update relationships in remaining blocks
    for block in new_blocks:
        if "Relationships" in block:
            updated_relationships = []
            for relationship in block["Relationships"]:
                if relationship["Type"] == "CHILD":
                    # Filter out removed IDs from CHILD relationships
                    relationship["Ids"] = [
                        child_id for child_id in relationship["Ids"] if child_id not in removed_ids
                    ]
                updated_relationships.append(relationship)
            block["Relationships"] = updated_relationships

    # Update the response with the cleaned blocks
    response["Blocks"] = new_blocks

    return response, removed

