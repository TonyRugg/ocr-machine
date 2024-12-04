import boto3
import time
from typing import Dict, List
import logging
import os
import trp.trp2 as t2
from trp.t_pipeline import add_page_orientation

def analyze_document(
    file_path: str = None,
    bucket_name: str = "textract-console-us-west-1-e2a6b2bf-ba3e-4c6d-a190-a725d00c82c4",
    document_key: str = None
) -> Dict:
    """
    Analyze a document using AWS Textract and enhance the response with page orientation data.
    """
    try:
        textract = boto3.client('textract', region_name='us-west-1')
        s3 = boto3.client('s3', region_name='us-west-1')
        
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
        
        # Process response with textract-response-parser to add orientation
        print("Adding page orientation information...")
        t_document = t2.TDocumentSchema().load(response)
        t_document = add_page_orientation(t_document)
        response = t2.TDocumentSchema().dump(t_document)
        
        return response
            
    except Exception as e:
        logging.error(f"Error analyzing document: {str(e)}")
        raise










__doc__ = """
AWS Textract Document Analysis with Page Orientation Enhancement

This script uses AWS Textract to analyze documents (PDF or image files) and enhance the response with page orientation data. 
It supports both synchronous processing for images and asynchronous processing for PDFs, leveraging S3 for document storage.

Dependencies:
    - boto3: For AWS Textract and S3 API interactions.
    - trp (Textract Response Parser): For processing and enhancing Textract responses with page orientation data.
    - logging: For error handling and reporting.

Key Function:
1. **analyze_document**:
    - Analyzes a document (PDF or image) using AWS Textract.
    - Handles both synchronous and asynchronous Textract APIs.
    - Enhances the response with page orientation information using the Textract Response Parser.

Arguments:
    - `file_path` (str): Path to the local file to analyze.
    - `bucket_name` (str): Name of the S3 bucket for storing PDF files. Default is a placeholder value.
    - `document_key` (str, optional): Key (name) for the file in S3. If not provided, the file's basename is used.

Returns:
    - `Dict`: Enhanced Textract response containing layout and table data, with added page orientation.

Implementation Details:
    - **PDF Files**:
        - Files are uploaded to the specified S3 bucket if not already present.
        - Asynchronous Textract API (`start_document_analysis`) is used for processing.
        - The script waits for the job to complete and fetches all blocks across multiple pages.
    - **Image Files**:
        - Image files are processed synchronously using `analyze_document`.
    - **Enhancements**:
        - The response is processed with the Textract Response Parser to add page orientation information.
        - Final output is a cleaned and enhanced JSON response.

Usage:
    - Import the `analyze_document` function:
        ```python
        from script_name import analyze_document
        ```
    - Call the function with the required arguments:
        ```python
        response = analyze_document(file_path="path/to/file.pdf")
        ```

Example:
    - Analyze a PDF file:
        ```python
        response = analyze_document(
            file_path="path/to/file.pdf",
            bucket_name="my-s3-bucket"
        )
        ```
    - Analyze an image file:
        ```python
        response = analyze_document(file_path="path/to/image.jpg")
        ```

Notes:
    - Ensure that AWS credentials are properly configured for the `boto3` client.
    - For PDF files, the S3 bucket must exist, and the user must have `s3:PutObject` permissions.
    - The script handles Textract's pagination by recursively fetching blocks with `NextToken`.

Error Handling:
    - Logs errors using the `logging` module.
    - Raises exceptions for missing input or failed Textract jobs.

"""
