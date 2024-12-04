import json
from PyPDF2 import PdfReader, PdfWriter
from .training_textract import analyze_document

FNAME = 'dallas-tx-3'
input_filepath = f"trainingSet/false_pdf/{FNAME}.pdf"
output_dir = "trainingSet/false_json/"
max_pages = 3000  # Maximum allowed pages per chunk

def split_pdf(input_pdf, max_pages):
    """Splits a PDF into chunks with a maximum number of pages."""
    reader = PdfReader(input_pdf)
    total_pages = len(reader.pages)
    chunks = []

    if total_pages <= max_pages:
        # No need for chunking, return the original file
        chunks.append(input_pdf)
    else:
        # Split into chunks
        for start_page in range(0, total_pages, max_pages):
            end_page = min(start_page + max_pages, total_pages)
            writer = PdfWriter()
            for page_num in range(start_page, end_page):
                writer.add_page(reader.pages[page_num])
            chunk_filename = f"{input_pdf[:-4]}_chunk_{start_page // max_pages + 1}.pdf"
            with open(chunk_filename, "wb") as chunk_file:
                writer.write(chunk_file)
            chunks.append(chunk_filename)

    return chunks

def process_chunks(chunks, output_dir):
    """Processes each PDF chunk with Textract and saves the response as JSON."""
    for chunk in chunks:
        chunk_name = chunk.split('/')[-1][:-4]
        response = analyze_document(file_path=chunk)
        with open(f"{output_dir}{chunk_name}.json", "w") as file:
            json.dump(response, file, indent=4)

if __name__ == "__main__":
    # Split the PDF into chunks
    pdf_chunks = split_pdf(input_filepath, max_pages)

    # Process each chunk and save the JSON response
    process_chunks(pdf_chunks, output_dir)

    print(f"Processed {len(pdf_chunks)} chunks. Results saved in {output_dir}.")








# Store the docstring in a variable
module_docstring = """
PDF Splitting and Processing Script

This script processes a PDF file by splitting it into smaller chunks if it exceeds a specified maximum number of pages. 
Each chunk (or the original file if no splitting is required) is then processed using Textract, and the results are saved 
as JSON files.

Modules Used:
    - json: For handling JSON data.
    - PyPDF2: For reading and writing PDF files.
    - training_textract: Custom module for analyzing document content with Textract.

Constants:
    - FNAME (str): Base name of the input PDF file.
    - input_filepath (str): Path to the input PDF file.
    - output_dir (str): Directory to save the JSON output files.
    - max_pages (int): Maximum allowed pages per chunk.

Functions:
    1. split_pdf(input_pdf, max_pages):
        Splits the input PDF file into smaller chunks if its page count exceeds `max_pages`.
        Returns a list of chunk filenames or the original filename if no splitting is required.

    2. process_chunks(chunks, output_dir):
        Processes each PDF chunk (or file) using Textract and saves the extracted data as a JSON file 
        in the specified output directory.

Usage:
    1. The script splits the input PDF file using `split_pdf`.
    2. Each resulting chunk (or the original file) is processed with `process_chunks`.
    3. JSON output is saved in the `output_dir`.

Output:
    - Processed JSON files for each chunk (or the entire file if no splitting is needed).
    - Console message summarizing the number of processed chunks and output directory.

Example:
    Run the script with the default parameters:
        `python script_name.py`

    Output:
        Processed 1 chunks. Results saved in trainingSet/false_json/.

Dependencies:
    - Textract SDK should be installed and configured.
    - The `training_textract` module must be available and provide the `analyze_document` function.
"""

# Assign the variable to the module's docstring
__doc__ = module_docstring

