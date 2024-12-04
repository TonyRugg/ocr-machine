import torch
from PIL import Image
import shutil
from pathlib import Path
from torchvision import transforms
import math

class StrikethroughTransform:
    def __init__(self, target_height=64, max_width=365):
        self.target_height = target_height
        self.max_width = max_width
        
    def __call__(self, image):
        # Convert to grayscale if not already
        if image.mode != 'L':
            image = image.convert('L')
            
        # First resize to target height while maintaining aspect ratio
        aspect_ratio = image.size[0] / image.size[1]
        new_width = int(self.target_height * aspect_ratio)
        image = image.resize((new_width, self.target_height), Image.Resampling.LANCZOS)
        
        # If width is greater than max_width, resize maintaining aspect ratio
        if new_width > self.max_width:
            image = image.resize((self.max_width, self.target_height), Image.Resampling.LANCZOS)
        # If width is less than max_width, repeat horizontally
        elif new_width < self.max_width:
            repeats = math.ceil(self.max_width / new_width)
            new_image = Image.new('L', (new_width * repeats, self.target_height), 255)
            for i in range(repeats):
                new_image.paste(image, (i * new_width, 0))
            # Crop to max_width if we exceeded it
            image = new_image.crop((0, 0, self.max_width, self.target_height))
            
        return transforms.ToTensor()(image)

class StrikethroughModel(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        from torchvision.models import mobilenet_v3_small
        
        self.mobilenet = mobilenet_v3_small(pretrained=False)
        
        # Modify first conv layer to accept single channel
        self.mobilenet.features[0][0] = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier for binary output
        self.mobilenet.classifier = torch.nn.Sequential(
            torch.nn.Linear(576, 1024),
            torch.nn.Hardswish(),
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(1024, 1),
            torch.nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.mobilenet(x).view(-1)

def process_images(input_dir, checkpoint_path, output_base_dir='sorted_images', batch_size=32):
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    true_dir = Path(output_base_dir) / 'true'
    false_dir = Path(output_base_dir) / 'false'
    true_dir.mkdir(parents=True, exist_ok=True)
    false_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = StrikethroughModel(pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Set up transform
    transform = transforms.Compose([
        StrikethroughTransform(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Get all image paths
    input_dir = Path(input_dir)
    image_paths = list(input_dir.rglob('*.jpeg'))  # Add more extensions if needed
    print(f"Found {len(image_paths)} images")
    
    # Process images
    processed_count = {'true': 0, 'false': 0}
    
    with torch.no_grad():
        for img_path in image_paths:
            try:
                # Load and transform image
                image = Image.open(img_path)
                image_tensor = transform(image).unsqueeze(0).to(device)
                
                # Get prediction
                output = model(image_tensor)
                is_strikethrough = output.item() >= 0.5
                
                # Copy to appropriate directory
                target_dir = true_dir if is_strikethrough else false_dir
                target_path = target_dir / img_path.name
                
                # Copy the file
                shutil.copy2(img_path, target_path)
                
                # Update counter
                key = 'true' if is_strikethrough else 'false'
                processed_count[key] += 1
                
                # Print progress
                if (processed_count['true'] + processed_count['false']) % 100 == 0:
                    print(f"Processed {processed_count['true'] + processed_count['false']} images")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Print final statistics
    print("\nProcessing complete!")
    print(f"Images with strikethrough: {processed_count['true']}")
    print(f"Images without strikethrough: {processed_count['false']}")
    print(f"Images are sorted into:")
    print(f"  - {true_dir}")
    print(f"  - {false_dir}")

if __name__ == "__main__":
    # Configuration
    config = {
        'input_dir': 'realSet/source_images/Table23-0471_residential_code',  # Directory containing images to process
        'checkpoint_path': 'checkpoints/best_model.pt',  # Path to your trained model
        'output_base_dir': 'sorted_images',  # Base directory for sorted images
        'batch_size': 64
    }
    
    # Run processing
    process_images(**config)