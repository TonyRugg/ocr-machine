import os
from PIL import Image
import torch
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import subprocess

def detect_tables(pdf_path, output_dir="output"):
    print(f"Starting table detection for {pdf_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    print("Model and processor loaded")
    
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    print(f"Converted PDF to {len(images)} images")
    
    for i, image in enumerate(images):
        print(f"\nProcessing page {i+1}")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        
        print(f"Found {len(results['boxes'])} potential tables")
        print(f"Scores: {results['scores'].tolist()}")
        
        for j, (box, score) in enumerate(zip(results["boxes"], results["scores"])):
            box = [round(i, 2) for i in box.tolist()]
            print(f"Table {j+1} box coordinates: {box}, confidence: {score:.2f}")
            if score > 0.7:
                table_img = image.crop(box)
                save_path = f"{output_dir}/page_{i+1}_table_{j+1}.jpg"
                table_img.save(save_path)
                print(f"Saved table to {save_path}")

if __name__ == "__main__":
    packages = [
        "torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu",
        "transformers",
        "pdf2image",
        "pillow==9.4.0"
    ]
    for pkg in packages:
        subprocess.run(f"pip install {pkg}", shell=True)
        
    pdf_path = input("Enter PDF file path: ")
    detect_tables(pdf_path)