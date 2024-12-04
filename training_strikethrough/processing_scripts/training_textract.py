import boto3
import time
from typing import Dict
import logging
import os

def analyze_document(
    file_path: str = None,
    bucket_name: str = "textract-console-us-west-1-e2a6b2bf-ba3e-4c6d-a190-a725d00c82c4",
    document_key: str = None
) -> Dict:
    """
    Extract just words from a document using AWS Textract.
    Returns a dict with only word blocks and their bounding boxes.
    """
    try:
        textract = boto3.client('textract', region_name='us-west-1')
        s3 = boto3.client('s3', region_name='us-west-1')
        
        # For files that need async processing (PDFs or S3 files)
        if (file_path and file_path.lower().endswith('.pdf')) or document_key:
            print("Processing document from S3...")
            
            # Upload file to S3 if file_path is provided
            if file_path and not document_key:
                document_key = os.path.basename(file_path)
                print(f"Uploading to S3: {bucket_name}/{document_key}")
                with open(file_path, 'rb') as document:
                    s3.upload_fileobj(document, bucket_name, document_key)
            
            # Start async job
            response = textract.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': bucket_name,
                        'Name': document_key
                    }
                }
            )
            
            job_id = response['JobId']
            print(f"Started job {job_id}")
            
            # Wait for completion
            while True:
                response = textract.get_document_text_detection(JobId=job_id)
                status = response['JobStatus']
                print(f"Status: {status}")
                if status in ['SUCCEEDED', 'FAILED']:
                    break
                time.sleep(5)
            
            if status == 'FAILED':
                raise Exception(f"Textract job failed: {response.get('StatusMessage', '')}")
            
            # Get all pages of words
            word_blocks = []
            pages_of_blocks = [response['Blocks']]
            next_token = response.get('NextToken')
            
            # Keep fetching while there's a NextToken
            while next_token:
                print("Fetching additional blocks...")
                response = textract.get_document_text_detection(
                    JobId=job_id,
                    NextToken=next_token
                )
                pages_of_blocks.append(response['Blocks'])
                next_token = response.get('NextToken')
            
            # Extract only WORD blocks from all pages and clean geometry
            for page in pages_of_blocks:
                for block in page:
                    if block['BlockType'] == 'WORD':
                        # Remove Polygon from Geometry
                        if 'Geometry' in block and 'Polygon' in block['Geometry']:
                            del block['Geometry']['Polygon']
                        word_blocks.append(block)
                        
        # For single-page image files
        else:
            if file_path:
                with open(file_path, 'rb') as document:
                    image_bytes = bytearray(document.read())
                response = textract.detect_document_text(
                    Document={'Bytes': image_bytes}
                )
                
                # Extract only WORD blocks and clean geometry
                word_blocks = []
                for block in response['Blocks']:
                    if block['BlockType'] == 'WORD':
                        # Remove Polygon from Geometry
                        if 'Geometry' in block and 'Polygon' in block['Geometry']:
                            del block['Geometry']['Polygon']
                        word_blocks.append(block)
            else:
                raise ValueError("Must provide either file_path or document_key")
        
        # Create final response with only word blocks
        final_response = {
            'Blocks': word_blocks,
            'DocumentMetadata': response['DocumentMetadata']
        }
        
        return final_response
            
    except Exception as e:
        logging.error(f"Error extracting text: {str(e)}")
        raise









__doc__ = """
AWS Textract Word Extraction Script

This script uses AWS Textract to extract only word blocks from documents, including their bounding box geometries.
It supports both synchronous and asynchronous processing, depending on the document type (image or PDF).

Dependencies:
    - boto3: For interacting with AWS Textract and S3.
    - logging: For error handling and reporting.
    - os: For file handling.

Function:
    analyze_document(file_path=None, bucket_name="...", document_key=None) -> Dict:
        Extracts words and their bounding boxes from a document using AWS Textract.

Arguments:
    - `file_path` (str, optional): Local path to the file to analyze. Supports PDF and image formats.
    - `bucket_name` (str, optional): Name of the S3 bucket to store PDFs for asynchronous processing. Default is a placeholder.
    - `document_key` (str, optional): Key (name) of the document in S3. If provided, skips file upload.

Returns:
    - `Dict`: A dictionary containing:
        - `Blocks`: A list of word blocks, each with bounding box geometry.
        - `DocumentMetadata`: Metadata about the processed document.

Implementation Details:
    - **PDF Files and S3 Objects**:
        - PDF files are uploaded to S3 and processed asynchronously using `start_document_text_detection`.
        - The script waits for the job to complete, handles pagination with `NextToken`, and extracts only word blocks.
    - **Image Files**:
        - Image files are processed synchronously using `detect_document_text`.
        - Word blocks are extracted directly from the response.
    - **Word Blocks**:
        - Only blocks of type `WORD` are included in the final response.
        - Polygon data in `Geometry` is removed to simplify output.

Usage:
    - Import the function and call it with appropriate arguments:
        ```python
        from script_name import analyze_document
        response = analyze_document(file_path="path/to/document.pdf")
        ```
    - For S3-hosted files:
        ```python
        response = analyze_document(bucket_name="my-s3-bucket", document_key="document.pdf")
        ```

Example:
    - Analyze a PDF file:
        ```python
        response = analyze_document(file_path="path/to/file.pdf", bucket_name="my-s3-bucket")
        ```
    - Analyze an image file:
        ```python
        response = analyze_document(file_path="path/to/image.jpg")
        ```

Notes:
    - Ensure that AWS credentials are properly configured for `boto3`.
    - For PDF files, the S3 bucket must exist, and the user must have `s3:PutObject` permissions.
    - The script handles Textract pagination for large documents with multiple pages.

Error Handling:
    - Logs errors using the `logging` module.
    - Raises exceptions for missing input or Textract job failures.

Output:
    - Returns a cleaned response containing only word blocks and their bounding boxes.

"""
