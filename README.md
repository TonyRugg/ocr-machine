
# Outline
- **Find Tables & Diagrams:** ID the Tables & Diagrams
- **Table/Diagram Parsing & Marking:** Parse the tables and diagrams into JSON and exclude them from the images for OCR
- **Outline Generation:** Generate Outline of headers through gpt-4o-mini by passing it the pdf image and raw text.  
	- If this exceeds the 128k token limit, including output, then chunk the document manually and provide chunks.
- **JSON Creation:** Pass the consolidated outline with the context of character range location and starting section for the chunk along with raw text and images and request a structured output that is the JSON outline of the document.

### Find Tables & Diagrams
- **gpt_table_detector.py** is used to detect a tables and diagrams in the pdf.
- gpt-4o-mini goes through the pages and finds all tables and diagrams and flags them in the final JSON 
- This JSON output includes the filepaths to the images

### Table/Diagram Parsing & Marking
- Perhaps clean and remove non-table/diagram pages manually before this processing step if unable to improve success rate
- Use AWS Textract through boto3 to boundary the tables and diagrams
- Have it read and export the raw text as well
- Process the outputs with Python tools to convert to digestible json 
- Create black boxes on the images and add text with unique identifiers for the diagrams and tables

### Outline Generation
- Provide the raw text as well as the original image pages back to gpt-4o-mini to generate an outline of Chapters and Sections
- Handle diagram and table references properly

### JSON Creation
- Provide the raw text and the outline to gpt-4o-mini to compile the final structured output json
- Use python to add in the diagrams and tables

---
# Handled Outside of Repo in new workflow
### JSON Chunking
- Chunk the text into sections and add all metadata for weaviate entry, most will be incomplete
- Flag the over token text to be split later

### Section Summarization
- Have gpt-4o or 4o1-mini process the document to: 
	- Summarize the sections (perhaps one step)
	- Generate category tags (subsequent steps)
	- Generate section reference tags
	- Generate defined terms list
	- If not already done, complete breadcrumb and other tags

### Embedding
- Connect the embed large model to weaviate and have all major properties vectorized on the way in

### Expansion projects
- Test a fine-tuned legalBERT or ollama model for efficacy in replacing some of the more expensive gpt steps in this process
