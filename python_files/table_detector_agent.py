import os
import base64
import openai
import json
from tabulate import tabulate
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pdf2image import convert_from_path
from tqdm import tqdm
from PyPDF2 import PdfReader, PdfWriter


def split_pdf_by_content(pdf_path, page_data):
    """
    Split PDF based on page content type (tables+diagrams/diagrams only/neither).
    Returns list of paths to created PDFs.
    """
    # Read the PDF
    reader = PdfReader(pdf_path)

    # Initialize writers
    table_writer = PdfWriter()
    diagram_writer = PdfWriter()
    other_writer = PdfWriter()

    # Extract page numbers for each category
    table_pages = set(page_data.get('table_pages', []))
    diagram_pages = set(page_data.get('diagram_pages', []))

    # Sort pages
    for i in range(len(reader.pages)):
        page = reader.pages[i]
        page_num = i + 1  # Convert 0-based index to 1-based page number

        if page_num in table_pages:
            table_writer.add_page(page)
        elif page_num in diagram_pages:
            diagram_writer.add_page(page)
        else:
            other_writer.add_page(page)

    # Save split PDFs
    output_files = []
    writers = [
        ('tables', table_writer),
        ('diagrams', diagram_writer),
        ('other', other_writer)
    ]

    for filename, writer in writers:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = Path(__file__).parent.parent / "intermediate_files" / f"{filename}_{timestamp}.pdf"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if len(writer.pages) > 0:
            with open(filepath, 'wb') as output:
                writer.write(output)
            output_files.append(str(filepath))

    return output_files



def process_page_with_gpt(image_base64, client, system_prompt, history_prompt, gpt_model):
    response = client.chat.completions.create(
        model=gpt_model,
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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "document_schema",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "table": {
                            "type": "string",
                            "description": "True if data table is present on page. False if not",
                            "enum": ['True', 'False']
                        },
                        "table_certainty": {
                            "type": ["string", "null"],
                            "description": "If True, give 0-1 rating and explanation of reasons why you chose yes"
                        },
                        "diagram": {
                            "type": "string", 
                            "description": "True if diagram is present on page. False if not",
                            "enum": ['True', 'False']
                        },
                        "diagram_certainty": {
                            "type": ["string", "null"],
                            "description": "If True, give 0-1 rating and explanation of reasons why you chose yes"
                        },
                    },
                    "required": ["table", "table_certainty", "diagram", "diagram_certainty"],
                    "additionalProperties": False
                }
            }
        },
        max_tokens=4000,
        temperature=0.1
    )
    return response


def encode_image(img_pth):
    with open(img_pth, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def process_json(response, page_num, image_path):
    page_json = response.choices[0].message.content
    page_dict = json.loads(page_json)
    page_dict = {key: (True if value == "True" else False if value == "False" else value) for key, value in page_dict.items()}
    page_dict['page_number'] = page_num
   
    if page_dict['table'] or page_dict['diagram']:
       page_dict['image_path'] = str(image_path)
    else:
       os.remove(image_path)
       
    return page_dict


def process_final_json(all_pages, output_path, total_tokens, pdf_file):
    # Get lists of pages where 'table' and 'diagram' are True
    table_pages = [page_num for page_num, page_data in all_pages.items() if page_data['table']]
    diagram_pages = [page_num for page_num, page_data in all_pages.items() if page_data['diagram']]
    
    final_dict = {
        "filename": pdf_file,
        "timestamp": datetime.now().isoformat(),
        "total_pages": len(all_pages),
        "table_pages": table_pages,  # Changed key and value
        "diagram_pages": diagram_pages,  # Changed key and value
        "total_tokens": total_tokens,
        "pages": all_pages
    }
    
    with open(output_path, 'w') as f:
        json.dump(final_dict, f, indent=4)
    
    return final_dict, len(table_pages), len(diagram_pages)


def detect_tables(input_pdf_path, gpt_model):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    total_count = prompt_count = completion_count = 0
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    openai.api_key = api_key
    system_prompt = """
    You are a precise document element detector in a highly critical environment. Your task is to: Report true if you detect a data table on the scanned page, false if not. Report true if you detect a technical diagram on the scanned page, else false. 
    Provide a 0-1 confidence and reasoning if you identify true for an item.  
    Just because you see indications of a diagram due to a reference or specifications does not mean there is one.  Be strict and confident before reporting true.
    Do not write anything else
    """
    history_prompt = "Please analyze this page."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json_path = Path(__file__).parent.parent / "intermediate_files" / f"detection_dict_{timestamp}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("Converting PDF to images...")
    images = convert_from_path(input_pdf_path, single_file=False, fmt='png', dpi=300)
    
    all_pages = {}
    pbar = tqdm(enumerate(images, start=1), total=len(images), desc="Processing pages")
    
    for page_num, image in pbar:
        image_path = Path(__file__).parent.parent / 'images' / f'page_{page_num}.png'
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(image_path)

        try:
            response = process_page_with_gpt(encode_image(image_path), openai, system_prompt, history_prompt, gpt_model)
            page_dict = process_json(response, page_num, image_path)
            all_pages[page_num] = page_dict
            
            total_tokens = float(response.usage.total_tokens)
            total_count += total_tokens
            prompt_count += float(response.usage.prompt_tokens)
            completion_count += float(response.usage.completion_tokens)
            
            pbar.set_postfix({'Total Tokens': f'{total_count:,.0f}'})

        except Exception as e:
            pbar.write(f"Error processing page {page_num}: {e}")

    response_dict, table_count, diagram_count = process_final_json(all_pages, output_json_path, total_count, input_pdf_path)

    print(f"\nProcessing complete. Dict saved as JSON to {output_json_path}.")

    table_data = [
        ["Completion", f'{completion_count:,.0f}'],
        ["Prompt", f'{prompt_count:,.0f}'],
        ["Total", f'{total_count:,.0f}'],
        ["Tables Found", f'{table_count}'],
        ["Diagrams Found", f'{diagram_count}']
    ]
    print(tabulate(table_data, headers=["Type", "Count"], tablefmt="grid"))

    pdf_list = split_pdf_by_content(input_pdf_path, response_dict)

    return pdf_list, response_dict
