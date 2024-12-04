
"""
STEP 1a: run table_detector_agent.detect_tables()
STEP 1b: process the resulting JSON with table_detector_agent.split_pdf_by_content() to get individual PDFs for each category 
STEP 2: Conditionally route to textract_agent.analyze_document() via textract_agent.textract_runner():
    - IF TABLE detected, route with TABLES and LAYOUT
    - IF Only DIAGRAM detected, route with LAYOUT
    - IF neither route with neither
STEP 3: Send the PDFs and their JSONs to white-out strikethroughs in both the PDFs and JSONs
STEP 4.1: Recombine them into a consolidated PDF and consolidated JSON with all white-out text removed
    * Will need to adjust page numbers to absolute page numbers when combining the JSONs 
STEP 4.2: Create a second version of the PDF , tables and diagrams whited out as well
STEP 4.3: Sumarization of tables and diagrams
    - Send the diagrams to the diagram summary gpt to receive summaries
    - Send the tables to the table summary
STEP 5: Add table and diagram summaries on top of whited out areas on PDF
STEP 6: Generate gpt outline of document based on whiteout PDF with summary plus JSON with summaries
    * This same gpt may be able to be used for the word doc/markdown versions of ordinances
STEP 7: Generate gpt complete text document with summaries still included in JSON.  Will need to pass the outline, source PDF and raw text to gpt
    * This may need to be chunked
STEP 8: Weaviate JSON generation: these will be core modules to be run through the chunking and embedding process later
    - Generate special weaviate JSONs for tables and diagrams from the textract JSON and the sumarizations
    - Generate final complete PDF JSON

*  All JSONs should be switched to dicts and json dumped to file periodically at milestone stages
* Prepare and test the gpt outliner with three flavors of document(1-all tables, 2-white-out tables, 3-white-out+summary tables)
* Are files being saved at every step in the event of failure, debug or retrenchment?
"""

import table_detector_agent as table_detector_agent
import textract_agent as textract_agent

PDF_FILENAME = 'Pages23-0471_residential_code.pdf'
GPT_MODEL = 'gpt-4o-mini'

pdf_filepath = f"source/{PDF_FILENAME}"

############# STEP 1 #############
pdf_list, detection_dict = table_detector_agent.detect_tables(pdf_filepath, GPT_MODEL)

############# STEP 2 #############
textract_responses_list = textract_agent.textract_runner(pdf_list, detection_dict, pdf_filepath)

############# STEP 3 #############


############# STEP 4 #############
