import os
from PIL import Image
import torch
import logging
from transformers import AutoImageProcessor, AutoModelForObjectDetection
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_diagrams(pdf_path, output_dir="output2"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Using a different model better suited for diverse image detection
    processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
    
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    
    for i, image in enumerate(images):
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]
        
        for j, (box, score, label) in enumerate(zip(results["boxes"], results["scores"], results["labels"])):
            category = model.config.id2label[label.item()]
            logger.info(f"Detected object: {category} with score {score:.3f}")
            
            box = [round(i, 2) for i in box.tolist()]
            try:
                diagram_img = image.crop(box)
                save_path = os.path.join(output_dir, f"page_{i+1}_{category}_{j+1}.jpg")
                diagram_img.save(save_path)
                logger.info(f"Saved {category} to {save_path}")
            except Exception as e:
                logger.error(f"Error saving image: {str(e)}")

if __name__ == "__main__":
    packages = [
        "torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "transformers",
        "pdf2image",
        "pillow"
    ]
    
    for pkg in packages:
        subprocess.run(f"pip install {pkg}", shell=True, check=True)
    
    pdf_path = input("Enter PDF file path: ")
    detect_diagrams(pdf_path)