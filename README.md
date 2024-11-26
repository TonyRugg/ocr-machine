
# Outline
1. **Raw Text OCR:** Get raw text through Document AI OCR
2. **Outline Generation:** Generate Outline of headers through gpt-4o-mini by passing it the pdf image and raw text.  
	- If this exceeds the 128k token limit, including output, then chunk the document manually and provide chunks.
3. **JSON Creation:** Pass the consolidated outline with the context of character range location and starting section for the chunk along with raw text and images and request a structured output that is the JSON outline of the document.

### Raw Text OCR


### Outline Generation


### JSON Creation
