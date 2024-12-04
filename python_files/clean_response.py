from detect_strikethrough import has_strikethrough
from trp import Document
from tqdm import tqdm

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











__doc__ = """
Clean Textract Responses with Strikethrough Detection

This script provides functionality to clean a Textract response by identifying and removing words 
with strikethrough marks in a PDF document. It uses the `has_strikethrough` function for detection 
and the `textract-response-parser` (`trp.Document`) for convenient handling of Textract response 
structures.

Dependencies:
    - detect_strikethrough: Custom module providing the `has_strikethrough` function.
    - textract-response-parser (`trp`): For parsing Textract responses.
    - tqdm: For progress visualization.

Functionality:
1. **clean_response_with_strikethrough**:
    - Processes a Textract response to detect and remove words with strikethrough.
    - Updates relationships in the response to ensure consistency after removing blocks.

Arguments:
    - `response` (dict): The Textract response dictionary.
    - `pdf_path` (str): Path to the PDF file used for detecting strikethrough.
    - `debug` (bool): Enables debug mode in the `has_strikethrough` function to save debug images.

Returns:
    - `tuple`:
        - `cleaned_response` (dict): The modified Textract response with strikethrough words removed.
        - `removed_bounding_boxes` (list): List of bounding boxes of the removed words.

Implementation Details:
- Converts the Textract response into a `Document` object for structured parsing.
- Iterates over the response "Blocks" and processes only those of type `WORD`.
- Uses the `has_strikethrough` function with bounding box geometry to determine if a word has strikethrough.
- Updates parent-child relationships in Textract "Blocks" to remove references to removed IDs.

Usage:
    - Import the `clean_response_with_strikethrough` function:
        ```python
        from script_name import clean_response_with_strikethrough
        ```
    - Call the function with the required arguments:
        ```python
        cleaned_response, removed_bboxes = clean_response_with_strikethrough(
            response=textract_response,
            pdf_path="path/to/pdf",
            debug=True
        )
        ```

Output:
    - A cleaned Textract response that excludes words with strikethrough.
    - A list of bounding boxes for the removed words, useful for debugging or visualization.

Notes:
- The `debug` mode in `has_strikethrough` saves diagnostic images for visualizing detected strikethroughs.
- Handles multi-page PDF documents and updates orientation-based geometry for each page.
"""
