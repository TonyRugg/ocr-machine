### Overview
This is a pipeline to generate a trainingset and train a mobileNet model to recognize strikethrough text.  

### Preprocessing - Data Generation
- First, word documents are selected and all strikethrough text is removed or added
- It is converted to PDF
- The PDF is run through the **fake_scan.py** to generate randomized new versions of the pdf to simulate scanner artifacts.
- The pdfs are then processed by running **run_textract.py** which calls the **training_textract.py**
    -  This generates a json with only the words and their bounding boxes.  all other boxtypes are removed as are the polygons since they take up space
- The json and the source pdf are fed to **create_bbox_chunked.py** to generate a new folder with the same filename and output all of the small word images. 
- Further distortions can be added, or can be done instead of **fake_scan.py**, by running the text images through **image_bad_scan.py**.

### Training
- Training is done using **model_training_pipeline.py** which handles all the pre-processing pipeline and training.  Training was done on an A4000 Paperspace machine using cuda.
- Inference can be handled to check and export to folders with **model_inference_test.py**
- Streaming inference can be handled using **streaming_strikethrough_detector** which will accept the image and return a bool

### Dataset Cold Storage
- All data for the training dataset has been archived in the following way:
    - The image files for most of the dataset were deleted
    - The json files were converted to mini jsons 
    - The image folders and pdfs were compressed using tar and then zstd
    - The entire traininSet was compressed using tar and zstd into a single object



##### Note
I am unable to share this publicly because of the confidential nature of the training set.  Perhaps a new training set could be made from public domain data at some point.