import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from PIL import Image
import numpy as np
from pathlib import Path
import math
from tqdm import tqdm
import time

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class StrikethroughDataset(Dataset):
    def __init__(self, true_dir, false_dir):
        self.transform = transforms.Compose([
            StrikethroughTransform(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Gather all image paths and labels
        self.data = []
        
        # Add true examples
        for img_path in Path(true_dir).rglob('*.jpeg'):
            self.data.append((str(img_path), 1.0))
            
        # Add false examples
        for img_path in Path(false_dir).rglob('*.jpeg'):
            self.data.append((str(img_path), 0.0))
            
        print(f"Found {len(self.data)} images total")
        print(f"True examples: {sum(1 for _, label in self.data if label == 1.0)}")
        print(f"False examples: {sum(1 for _, label in self.data if label == 0.0)}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path)
            
            if self.transform:
                image = self.transform(image)
                
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image and the label if there's an error
            return torch.zeros((1, self.transform.transforms[0].target_height, 
                              self.transform.transforms[0].max_width)), torch.tensor(label, dtype=torch.float32)

def create_data_loaders(true_dir, false_dir, batch_size=32, train_split=0.7, 
                       val_split=0.15, test_split=0.15, num_workers=4, seed=42):
    """
    Creates train, validation, and test data loaders from true/false directories
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create dataset
    dataset = StrikethroughDataset(true_dir, false_dir)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(train_dataset.indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_dataset.indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(test_dataset.indices),
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

class StrikethroughModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        # Load pretrained model
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.mobilenet = mobilenet_v3_small(weights=weights)
        
        # Modify first conv layer to accept single channel
        self.mobilenet.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        
        # Modify classifier for binary output
        self.mobilenet.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.mobilenet(x).view(-1)

def train_model(true_dir, false_dir, num_epochs=10, batch_size=32, learning_rate=0.001, 
                weight_decay=1e-5, num_workers=4, checkpoint_dir='checkpoints'):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        true_dir=true_dir,
        false_dir=false_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model
    model = StrikethroughModel(pretrained=True).to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_losses.append(loss.item())
            predictions = (outputs >= 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{np.mean(train_losses):.4f}', 
                            'acc': f'{100 * train_correct/train_total:.2f}%'})
        
        # Validation phase
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_losses.append(loss.item())
                predictions = (outputs >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)
        
        # Calculate epoch statistics
        train_loss = np.mean(train_losses)
        train_acc = 100 * train_correct / train_total
        val_loss = np.mean(val_losses)
        val_acc = 100 * val_correct / val_total
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, f'{checkpoint_dir}/best_model.pt')
    
    # Final test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            predictions = (outputs >= 0.5).float()
            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100 * test_correct / test_total
    print(f'\nFinal Test Accuracy: {test_acc:.2f}%')
    
    return model

def load_trained_model(checkpoint_path):
    model = StrikethroughModel(pretrained=False)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == "__main__":
    # Training configuration
    config = {
        'true_dir': 'trainingSet/true_images',
        'false_dir': 'trainingSet/false_images',
        'num_epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_workers': 8,
        'checkpoint_dir': 'checkpoints'
    }
    
    # Train the model
    model = train_model(**config)










__doc__ = """
Strikethrough Detection Pipeline

This script provides a complete pipeline for training, validating, and testing a neural network model for detecting 
strikethrough patterns in images. It uses PyTorch and Torchvision for deep learning, along with custom preprocessing 
and dataset handling.

Dependencies:
    - PyTorch: For neural network implementation and training.
    - Torchvision: For pretrained models and image transformations.
    - Pillow: For image processing.
    - NumPy: For numerical computations.
    - tqdm: For progress bars.

Key Components:
1. **StrikethroughTransform**:
    - A custom image preprocessing class that resizes images to a fixed height while maintaining the aspect ratio, 
      and adjusts the width to fit a specified maximum.
    - Converts the image to a tensor for PyTorch compatibility.

2. **StrikethroughDataset**:
    - A PyTorch `Dataset` for loading images from two directories (true and false examples).
    - Applies preprocessing and normalization.
    - Handles corrupted or missing files gracefully.

3. **StrikethroughModel**:
    - A neural network based on MobileNetV3, modified for single-channel grayscale input and binary classification output.
    - Uses sigmoid activation for binary probability prediction.

4. **Data Loading**:
    - `create_data_loaders` splits the dataset into training, validation, and testing sets.
    - Utilizes `DataLoader` for batching and shuffling.

5. **Training and Evaluation**:
    - `train_model` trains the model, tracks performance on validation data, and saves the best checkpoint.
    - Outputs loss and accuracy metrics for each epoch.

6. **Model Loading**:
    - `load_trained_model` allows for loading a pretrained model from a saved checkpoint.

Execution:
    - The script is executable as a standalone program.
    - When executed, it trains the model on data from specified directories and evaluates it on a test set.

Example Usage:
    - Training:
        ```
        python script_name.py
        ```
        Adjust the configuration in the `if __name__ == "__main__"` block to set paths and parameters.
    - Loading a Trained Model:
        ```
        from script_name import load_trained_model
        model = load_trained_model('checkpoints/best_model.pt')
        ```

Configuration:
    - Default configuration can be modified in the `if __name__ == "__main__"` block, including:
        - Directory paths for training data.
        - Hyperparameters such as learning rate, batch size, and number of epochs.
        - Checkpoint directory for saving the best model.

Outputs:
    - Training progress with loss and accuracy metrics.
    - Validation and testing performance.
    - Saved model checkpoints.

"""
