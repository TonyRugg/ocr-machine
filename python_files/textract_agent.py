import boto3
import time
import json
from typing import Dict, List
import logging
import os
import trp.trp2 as t2
from datetime import datetime
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from trp.t_pipeline import add_page_orientation

def analyze_document(
    file_path: str = None,
    bucket_name: str = "textract-console-us-west-1-e2a6b2bf-ba3e-4c6d-a190-a725d00c82c4",
    document_key: str = None,
    features=None
) -> Dict:
    """
    Analyze a document using AWS Textract and enhance the response with page orientation data.
    """
    try:
        textract = boto3.client('textract', region_name='us-west-1')
        s3 = boto3.client('s3', region_name='us-west-1')

        # Ensure the file path is a string
        if isinstance(file_path, Path):
            file_path = str(file_path)

        # For PDF files, use the async API
        if file_path and file_path.lower().endswith('.pdf'):
            print("Processing PDF file...")
            
            # Upload file to S3 if not already there
            if not document_key:
                document_key = os.path.basename(file_path)

            print(f"Uploading to S3: {bucket_name}/{document_key}")
            with open(file_path, 'rb') as document:
                s3.upload_fileobj(document, bucket_name, document_key)

            # Start async job
            if features:
                print(f'Requesting Textract analysis of {features}')
                response = textract.start_document_analysis(
                    DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_key}},
                    FeatureTypes=features
                )
                job_type = "ANALYSIS"
            else:
                print('No features detecting. Requesting Textract only perform text extraction')
                response = textract.start_document_text_detection(
                    DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': document_key}}
                )
                job_type = "TEXT_DETECTION"

            job_id = response['JobId']
            print(f"Started job {job_id}")

            # Wait for completion
            while True:
                if job_type == "ANALYSIS":
                    response = textract.get_document_analysis(JobId=job_id)
                else:  # job_type == "TEXT_DETECTION"
                    response = textract.get_document_text_detection(JobId=job_id)

                status = response['JobStatus']
                print(f"Status: {status}")
                if status in ['SUCCEEDED', 'FAILED']:
                    break
                time.sleep(5)

            if status == 'SUCCEEDED':
                # Get the initial response
                pages_of_blocks = [response['Blocks']]
                next_token = response.get('NextToken')

                # Keep fetching while there's a NextToken
                while next_token:
                    print("Fetching additional blocks...")
                    if job_type == "ANALYSIS":
                        response = textract.get_document_analysis(
                            JobId=job_id,
                            NextToken=next_token
                        )
                    else:  # job_type == "TEXT_DETECTION"
                        response = textract.get_document_text_detection(
                            JobId=job_id,
                            NextToken=next_token
                        )

                    pages_of_blocks.append(response['Blocks'])
                    next_token = response.get('NextToken')

                # Combine all blocks
                response['Blocks'] = [block for page in pages_of_blocks for block in page]
                # Remove NextToken since we've handled it
                response.pop('NextToken', None)

            else:
                raise ValueError(f"Textract job failed with status: {status}")

        else:
            raise ValueError("Must provide file_path for non-PDF documents")

        # Process response with textract-response-parser to add orientation
        print("Adding page orientation information...")
        t_document = t2.TDocumentSchema().load(response)
        t_document = add_page_orientation(t_document)
        response = t2.TDocumentSchema().dump(t_document)

        return response

    except Exception as e:
        logging.error(f"Error analyzing document: {str(e)}")
        raise


def recombine_split_results(pdf_path_list: List[str], responses: List[Dict], original_response_dict: Dict, original_pdf_path: str):
    """
    Recombine split PDFs into a single PDF with their original page numbers
    and consolidate the dictionaries from Textract responses.

    Parameters:
    - pdf_path_list: List of file paths to split PDFs.
    - responses: List of Textract responses with metadata.
    - original_response_dict: Original response dict with table_pages and diagram_pages keys.
    - original_pdf_path: Path to the original PDF.

    Returns:
    - A tuple (consolidated_pdf_path, consolidated_dict):
      - consolidated_pdf_path: Path to the recombined PDF.
      - consolidated_dict: Dictionary with consolidated Textract data.
    """
    try:
        # Prepare consolidated PDF and dictionary
        writer = PdfWriter()
        consolidated_dict = {"filename": original_pdf_path, "pages": {}}

        # Map split PDFs to their respective page groups
        table_pdf_path = next((p for p in pdf_path_list if "tables" in str(p).lower()), None)
        diagram_pdf_path = next((p for p in pdf_path_list if "diagrams" in str(p).lower()), None)
        other_pdf_path = next((p for p in pdf_path_list if "other" in str(p).lower()), None)

        table_pages = original_response_dict.get("table_pages", [])
        diagram_pages = original_response_dict.get("diagram_pages", [])
        other_pages = [
            p for p in range(1, original_response_dict["total_pages"] + 1)
            if p not in table_pages and p not in diagram_pages
        ]

        # Maintain the order of all pages based on the original document
        all_pages_ordered = sorted(table_pages + diagram_pages + other_pages)

        # Load readers for split PDFs with explicit mapping
        pdf_readers = {
            "tables": (PdfReader(table_pdf_path), table_pages) if table_pdf_path else None,
            "diagrams": (PdfReader(diagram_pdf_path), diagram_pages) if diagram_pdf_path else None,
            "other": (PdfReader(other_pdf_path), other_pages) if other_pdf_path else None,
        }

        # Combine pages in the correct order
        for page_num in all_pages_ordered:
            # Determine the source of the page (tables, diagrams, or other)
            for pdf_type, (reader, pages) in pdf_readers.items():
                if page_num in pages:
                    page_index = pages.index(page_num)  # Index within the split PDF
                    writer.add_page(reader.pages[page_index])

                    # Match the file path from pdf_path_list
                    matching_path = next((p for p in pdf_path_list if pdf_type in str(p).lower()), None)
                    if matching_path is None:
                        raise ValueError(f"No matching path found for {pdf_type}")

                    # Add the Textract data for this page
                    response_index = pdf_path_list.index(matching_path)
                    textract_data = responses[response_index].get("response", {}).get("Blocks", [])
                    consolidated_dict["pages"][page_num] = textract_data
                    break

        # Save the consolidated PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        consolidated_pdf_path = Path(__file__).parent.parent / "intermediate_files" / f"consolidated_{timestamp}.pdf"
        consolidated_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(consolidated_pdf_path, 'wb') as f:
            writer.write(f)

        # Save consolidated dictionary to intermediate_files
        consolidated_json_path = Path(__file__).parent.parent / "intermediate_files" / f"consolidated_{timestamp}.json"
        with open(consolidated_json_path, 'w') as f:
            json.dump(consolidated_dict, f, indent=4)

        print(f"Consolidated PDF saved to {consolidated_pdf_path}")
        print(f"Consolidated dictionary saved to {consolidated_json_path}")

        return str(consolidated_pdf_path), consolidated_dict

    except Exception as e:
        logging.error(f"Error recombining split results: {str(e)}")
        raise



def textract_runner(pdf_path_list, detection_dict, original_pdf_path):
    """
    Read the name of each PDF from the file path and call analyze_document with the correct arguments.
    Compile a list of responses and return.
    """
    responses = []

    for pdf_path in pdf_path_list:
        features = None  # Text detection only
        try:
            # Determine the type of file from the name
            if "tables" in str(pdf_path).lower():
                features = ["TABLES", "LAYOUT"]
                print(f"Processing {pdf_path} with features: {features}")
            elif "diagrams" in str(pdf_path).lower():
                features = ["LAYOUT"]
                print(f"Processing {pdf_path} with features: {features}")
            else:
                print(f"Processing {pdf_path} with text detection")


            # Call analyze_document for each file
            response = analyze_document(file_path=pdf_path, features=features)
            
            # Save response as JSON
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(pdf_path).stem  # Extract base name without extension
            json_path = Path(__file__).parent.parent / "intermediate_files" / f"{filename}_{timestamp}.json"
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w') as f:
                json.dump(response, f, indent=4)
            
            print(f"Saved response to {json_path}")

            # Append the response to the list
            responses.append({
                "file_path": pdf_path,
                "features": features or "Text Detection",
                "response": response
            })

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            responses.append({
                "file_path": pdf_path,
                "features": features or "Text Detection",
                "error": str(e)
            })
    
    consolidated_pdf_path, consolidated_dict = recombine_split_results(pdf_path_list, responses, detection_dict, original_pdf_path)

    return consolidated_pdf_path, consolidated_dict

