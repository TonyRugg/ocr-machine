import os
import base64
import openai
import json
from PIL import Image
from io import BytesIO
from tabulate import tabulate
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from pdf2image import convert_from_path
from tqdm import tqdm

PDF_FILE = 'Pages23-0471_residential_code.pdf'
GPT_MODEL = 'gpt-4o-mini'


def send_table_outside(page_num, image_path):
    # print(f"Sending table from page {page_num} for external processing")
    return True

def process_page_with_gpt(image_base64, client, system_prompt, history_prompt):
    response = client.chat.completions.create(
        model=GPT_MODEL,
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

def process_json(response, page_num, image_path, pdf_file):
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
   table_count = sum(1 for page in all_pages.values() if page['table'])
   diagram_count = sum(1 for page in all_pages.values() if page['diagram'])
   
   final_json = {
       "filename": pdf_file,
       "timestamp": datetime.now().isoformat(),
       "total_pages": len(all_pages),
       "total_tables": table_count,
       "total_diagrams": diagram_count,
       "total_tokens": total_tokens,
       "pages": all_pages
   }
   
   with open(output_path, 'w') as f:
       json.dump(final_json, f, indent=4)
   
   return table_count, diagram_count

def main(input_pdf_path):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    total_count = prompt_count = completion_count = 0
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found")
    
    openai.api_key = api_key
    system_prompt = """
    You are a precise document element detector. Your task is to: Report true if you detect a data table on the scanned page, false if not report true if you detect a technical diagram on the scanned page, false. 
    Provide a 0-1 confidence and reasoning if you identify true for an item.  
    Just because you see indications of a diagram due to a reference or specifications does not mean there is one.  Be strict and confident before reporting true.  
    Do not write anything else
    """
    history_prompt = "Please process this document"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_txt_path = Path(__file__).parent.parent / "raw_text" / f"text_{timestamp}.json"

    print("Converting PDF to images...")
    images = convert_from_path(input_pdf_path, single_file=False, fmt='png', dpi=300)
    
    all_pages = {}
    pbar = tqdm(enumerate(images, start=1), total=len(images), desc="Processing pages")
    
    for page_num, image in pbar:
        image_path = Path(__file__).parent.parent / 'images' / f'page_{page_num}.png'
        image.save(image_path)

        try:
            response = process_page_with_gpt(encode_image(image_path), openai, system_prompt, history_prompt)
            page_dict = process_json(response, page_num, image_path, PDF_FILE)
            all_pages[page_num] = page_dict
            
            total_tokens = float(response.usage.total_tokens)
            total_count += total_tokens
            prompt_count += float(response.usage.prompt_tokens)
            completion_count += float(response.usage.completion_tokens)
            
            pbar.set_postfix({'Total Tokens': f'{total_count:,.0f}'})

        except Exception as e:
            pbar.write(f"Error processing page {page_num}: {e}")

    table_count, diagram_count = process_final_json(all_pages, output_txt_path, total_count, PDF_FILE)
   
    print(f"\nProcessing complete. Text saved to {output_txt_path}.")

    table_data = [
        ["Completion", f'{completion_count:,.0f}'],
        ["Prompt", f'{prompt_count:,.0f}'],
        ["Total", f'{total_count:,.0f}'],
        ["Tables Found", f'{table_count}'],
        ["Diagrams Found", f'{diagram_count}']
    ]
    print(tabulate(table_data, headers=["Type", "Count"], tablefmt="grid"))

if __name__ == "__main__":
    input_pdf_path = f"source/{PDF_FILE}"
    main(input_pdf_path)