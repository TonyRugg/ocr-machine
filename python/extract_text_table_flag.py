import os
import base64
import openai

from PIL import Image
from io import BytesIO
from tabulate import tabulate
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pdf2image import convert_from_path
# os.system('clear')


TEXT_FOLDER = "raw_text"

def process_page_with_gpt(image_base64, client, system_prompt, history_prompt):
    # Process a page image using GPT to extract text, format tables as Markdown, 
    # and reference images or diagrams with unique descriptive names.
    # Args: image_base64 (str): Base64 encoded image.
    #       client: OpenAI client instance.
    #       system_prompt (str): System prompt for the GPT model.
    #       history_prompt (str): History/context prompt for the GPT model.
    # Returns: dict: The full response from GPT.

    # # Image override testing
    # image = Image.new("RGB", (1, 1), (0, 0, 0))
    # buffer = BytesIO()
    # image.save(buffer, format="PNG")  # Save as PNG format
    # buffer.seek(0)  # Move to the start of the buffer
    # image_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": history_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=4000,
        temperature=0.1
    )
    return response

def encode_image(img_pth):
    with open(img_pth, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8') 

def resize_image(image):
    """
    Resizes the input image to 512x512 pixels, maintaining aspect ratio and adding padding,
    to simulate GPT-4o-mini's low-resolution mode processing.

    Parameters:
        image (PIL.Image.Image): The input image.

    Returns:
        PIL.Image.Image: The resized image.
    """
    target_size = (512, 512)
    image_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]

    # Determine the new size while maintaining aspect ratio
    if image_ratio > target_ratio:
        # Image is wider than target aspect ratio
        new_width = target_size[0]
        new_height = int(target_size[0] / image_ratio)
    else:
        # Image is taller than target aspect ratio
        new_height = target_size[1]
        new_width = int(target_size[1] * image_ratio)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with white background
    new_image = Image.new("RGB", target_size, (255, 255, 255))

    # Paste the resized image onto the center of the new image
    paste_position = ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2)
    new_image.paste(resized_image, paste_position)

    return new_image

def extract_text_and_metadata(response):
    # Extract raw text and table presence flag from the GPT response.
    # Args:response: Full GPT response for a page.
    # Returns:tuple: Raw text and a boolean indicating if a table is present.
    
    content = response.choices[0].message.content.strip()
    # Look for "Table Present: True" or "Table Present: False" in the content
    lines = content.splitlines()
    table_present = False
    extracted_text = []
    for line in lines:
        if line.lower().startswith("table present:"):
            table_present = "true" in line.lower()
        else:
            extracted_text.append(line)
    return "\n".join(extracted_text).strip(), table_present

def main(input_pdf_path):
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    openai.api_key = api_key

    # System and history prompts
    # system_prompt = (
    #     "You are a precise legal document processor. "
    #     "For each page, extract text and identify any tables and diagrams with in-line TableFound and DiagramFound flags. "
    #     "If there are multiple diagrams or multiple tables close to one another interperet as one. "
    #     "Replace the diagram and/or table with a unique filename ending in '.png' and generate a bounding box for each, formatted as x1,y1,x2,y2 "
    #     "where coordinates are pixels from top-left of page. "
    #     "(Format Example 'TableFound: Table_504_4.png (bbox: 50, 200, 700, 600)') "
    #     "Make sure this reference is in-line in the exact text locaiton where the item was removed"
    # )

    system_prompt = (
    "You are a precise document element detector. Your task is to: "
    "1. First output the exact image dimensions you see in this format: "
    "'IMAGE_SIZE: width,height' "
    "2. Then locate ONLY tables with grid lines/cells/borders. "
    "Output coordinates ONLY for the bordered table structure in this format: "
    "'TABLE_COORDS: x1,y1,x2,y2' where: "
    "- Coordinates are measured in pixels from the document's top-left (0,0) "
    "- Include ONLY the table with grid lines and its immediate title row "
    "- Exclude all regular paragraph text, even if related to the table "
    "Output ONLY these coordinate lines, no other descriptions or text"
    )

    history_prompt = (
    "Please first report the image dimensions you see, then provide the exact pixel coordinates "
    "of any tables, measured from the top-left corner (0,0). Use only the formats "
    "'IMAGE_SIZE: x,y' and 'TABLE_COORDS: x1,y1,x2,y2' with no other text."
    )

    # Add a datetime stamp to the output filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt_path = Path(__file__).parent.parent / TEXT_FOLDER /f"text_{timestamp}.txt"

    # Convert PDF to images
    print("Converting PDF to images...")
    images = convert_from_path(input_pdf_path, single_file=False, fmt='png', dpi=300)

    concatenated_text = ""
    all_responses = {}
    
    # Process each image
    for page_num, image in enumerate(images, start=1):
        print(f"Processing page {page_num}...")
        
        image_path = Path(__file__).parent.parent / 'images' / f'page_{page_num}.png'
        image_path_low = Path(__file__).parent.parent / 'images' / f'low_page_{page_num}.png'
        print(image.size)
        image.save(image_path)

        image_low = resize_image(image)
        image_low.save(image_path_low)
        print(image_low.size)
        image_base64 = encode_image(image_path)
        
        try:
            response = process_page_with_gpt(image_base64, openai, system_prompt, history_prompt)
            raw_text, table_present = extract_text_and_metadata(response)
            
            # Create table data
            table_data = [
                ["Completion", f'{response.usage.completion_tokens:,.0f}'],
                ["Prompt", f'{response.usage.prompt_tokens:,.0f}'],
                ["Total", f'{response.usage.total_tokens:,.0f}'],
            ]
            
            # Print GPT-provided table presence flag
            print(f"Page {page_num}: Contains table = {table_present}")
            print(tabulate(table_data, headers=["Token Type", "Count"], tablefmt="grid"))
            # Append the raw text to the concatenated output
            concatenated_text += f"\n--- Page {page_num} ---\n{raw_text}\n"
            
            # Store the full response
            all_responses[page_num] = response
        
        except Exception as e:
            print(f"Error processing page {page_num}: {e}")

    # Save the concatenated text to the output file
    with open(output_txt_path, "w") as output_file:
        output_file.write(concatenated_text)

    
    
    print(f"Processing complete. Text saved to {output_txt_path}.")

if __name__ == "__main__":
    # Input file path
    input_pdf_path = "source/Pages23-0468_building_code.pdf"  # Replace with your input PDF path
    
    main(input_pdf_path)
