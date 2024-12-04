import boto3
import time
from typing import Dict, List
import logging
import os

def analyze_document(
    file_path: str = None,
    bucket_name: str = "textract-console-us-west-1-e2a6b2bf-ba3e-4c6d-a190-a725d00c82c4",
    document_key: str = None
) -> Dict:
    """
    Analyze a document using AWS Textract.
    """
    try:
        textract = boto3.client('textract', region_name='us-west-1')  # Changed region to match bucket
        s3 = boto3.client('s3', region_name='us-west-1')
        
        # For PDF files, we must use the async API
        if file_path and file_path.lower().endswith('.pdf'):
            print("Processing PDF file...")
            
            # Upload file to S3 if not already there
            if not document_key:
                document_key = os.path.basename(file_path)
            
            print(f"Uploading to S3: {bucket_name}/{document_key}")
            with open(file_path, 'rb') as document:
                s3.upload_fileobj(document, bucket_name, document_key)
            
            # Start async job with correct parameters
            response = textract.start_document_analysis(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': document_key
                    }
                },
                FeatureTypes=['LAYOUT', 'TABLES'],
            )
            
            job_id = response['JobId']
            print(f"Started job {job_id}")
            
            # Wait for completion
            while True:
                response = textract.get_document_analysis(JobId=job_id)
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
                    response = textract.get_document_analysis(
                        JobId=job_id,
                        NextToken=next_token
                    )
                    pages_of_blocks.append(response['Blocks'])
                    next_token = response.get('NextToken')
                
                # Combine all blocks
                response['Blocks'] = [block for page in pages_of_blocks for block in page]
                # Remove NextToken since we've handled it
                response.pop('NextToken', None)
                return response
                
        # For non-PDF files
        else:
            if file_path:
                with open(file_path, 'rb') as document:
                    image_bytes = bytearray(document.read())
                response = textract.analyze_document(
                    Document={'Bytes': image_bytes},
                    FeatureTypes=['LAYOUT', 'TABLES']
                )
            else:
                raise ValueError("Must provide file_path for non-PDF documents")
                
            return response
            
    except Exception as e:
        logging.error(f"Error analyzing document: {str(e)}")
        raise