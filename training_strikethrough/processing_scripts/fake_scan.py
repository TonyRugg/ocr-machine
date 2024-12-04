import numpy as np
from PIL import Image
import cv2
from pdf2image import convert_from_path
import random
from datetime import datetime
import img2pdf
import os
from tqdm import tqdm
from PyPDF2 import PdfWriter, PdfReader
import io

class ScannerParams:
    def __init__(self):
        # Document-level parameters
        self.noise_level = random.uniform(0.05, 0.1)      # Background noise
        self.blur_radius = random.uniform(0.6, 0.9)       # Increased blur for text
        self.contrast = random.uniform(1.15, 1.25)        # Increased contrast for text edges
        self.brightness = random.uniform(-0.05, 0.05)     # Reduced brightness variation
        self.rotation = random.uniform(-0.3, 0.3)
        self.jpeg_quality = random.randint(82, 88)        # Lower quality for more text artifacts
        self.dust_particles = random.randint(15, 25)      # More dust
        self.scratches = random.randint(2, 5)             # Random scratches
        self.hairs = random.randint(1, 3)                 # Random hairs
        self.light_variation = random.uniform(0.995, 1.005)
        self.color_tint = (
            random.uniform(0.995, 1.005),
            random.uniform(0.995, 1.005),
            random.uniform(0.995, 1.000)
        )
        self.edge_threshold = random.uniform(40, 80)      # Lower threshold to catch more text edges
        self.edge_noise = random.uniform(0.2, 0.4)        # More noise around text edges
    
    def print_params(self):
        """Print all parameters in a formatted way"""
        print("\nScanner Parameters:")
        print(f"├── Noise Level: {self.noise_level:.3f} (0.05-0.1, higher = more noise)")
        print(f"├── Blur Radius: {self.blur_radius:.3f} (0.6-0.9, higher = more blur)")
        print(f"├── Contrast: {self.contrast:.3f} (1.15-1.25)")
        print(f"├── Brightness: {self.brightness:.3f} (-0.05 to +0.05)")
        print(f"├── Rotation: {self.rotation:.3f}° (-0.3° to +0.3°)")
        print(f"├── JPEG Quality: {self.jpeg_quality} (82-88)")
        print(f"├── Dust Particles: {self.dust_particles} (15-25)")
        print(f"├── Scratches: {self.scratches} (2-5)")
        print(f"├── Hairs: {self.hairs} (1-3)")
        print(f"├── Edge Noise: {self.edge_noise:.3f} (0.2-0.4)")
        print(f"├── Light Variation: {self.light_variation:.3f} (0.995-1.005)")
        print(f"└── Color Tint (RGB):")
        print(f"    ├── Red: {self.color_tint[0]:.3f}")
        print(f"    ├── Green: {self.color_tint[1]:.3f}")
        print(f"    └── Blue: {self.color_tint[2]:.3f}")

def add_text_noise(img_array, params):
    """Add noise specifically to text areas"""
    # Convert to grayscale to identify dark areas (text)
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Create mask for dark areas (text)
    _, text_mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Add noise specifically to text areas
    text_noise = np.random.normal(0, 30, img_array.shape).astype(np.uint8)
    text_noise = cv2.GaussianBlur(text_noise, (3, 3), 0.5)
    
    # Scale noise by text mask and ensure same data type
    text_mask_3d = np.stack([text_mask] * 3, axis=-1) / 255.0
    text_noise_effect = (text_noise * text_mask_3d).astype(np.uint8)
    
    # Apply noise to image
    result = img_array.copy()
    cv2.addWeighted(img_array, 1, text_noise_effect, 0.3, 0, result)
    
    # Add some random holes in text
    holes = np.random.random(img_array.shape[:2]) > 0.998
    holes = holes.astype(np.uint8) * 255
    holes_3d = np.stack([holes] * 3, axis=-1)
    
    # Only apply holes to text areas
    holes_in_text = cv2.bitwise_and(holes_3d, holes_3d, mask=text_mask)
    cv2.add(result, holes_in_text, result)
    
    return result

def add_dust_and_artifacts(img_array, params):
    """Add realistic scanner artifacts like dust, scratches, and hairs"""
    height, width = img_array.shape[:2]
    result = img_array.copy()
    
    # Add dust particles with varying sizes and intensities
    for _ in range(params.dust_particles):
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        radius = random.randint(1, 3)
        color = random.randint(180, 255)
        opacity = random.uniform(0.3, 1.0)
        
        dust = np.zeros_like(img_array)
        cv2.circle(dust, (x, y), radius, (color, color, color), -1)
        cv2.addWeighted(result, 1, dust.astype(np.uint8), opacity, 0, result)
    
    # Add scratches
    for _ in range(params.scratches):
        start_x = random.randint(0, width-1)
        start_y = random.randint(0, height-1)
        length = random.randint(50, 200)
        angle = random.uniform(0, 360)
        thickness = random.randint(1, 2)
        color = random.randint(200, 255)
        opacity = random.uniform(0.2, 0.7)
        
        end_x = int(start_x + length * np.cos(np.radians(angle)))
        end_y = int(start_y + length * np.sin(np.radians(angle)))
        
        scratch = np.zeros_like(img_array)
        cv2.line(scratch, (start_x, start_y), (end_x, end_y), 
                (color, color, color), thickness)
        cv2.addWeighted(result, 1, scratch.astype(np.uint8), opacity, 0, result)
    
    # Add hairs
    for _ in range(params.hairs):
        points = []
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        
        # Generate curved hair using multiple points
        num_points = random.randint(5, 10)
        for i in range(num_points):
            x += random.randint(-20, 20)
            y += random.randint(-20, 20)
            x = np.clip(x, 0, width-1)
            y = np.clip(y, 0, height-1)
            points.append((x, y))
        
        # Draw curved hair
        if len(points) >= 2:
            hair = np.zeros_like(img_array)
            color = random.randint(180, 230)
            opacity = random.uniform(0.3, 0.6)
            
            cv2.polylines(hair, [np.array(points)], False, 
                         (color, color, color), 1, cv2.LINE_AA)
            cv2.addWeighted(result, 1, hair.astype(np.uint8), opacity, 0, result)
    
    return result

def degrade_text_edges(img_array, params):
    """Apply enhanced degradation to text edges"""
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Detect edges with lower threshold to catch more text
    edges = cv2.Canny(gray, params.edge_threshold, params.edge_threshold * 2)
    
    # Dilate edges slightly
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Add more intense noise to edge areas
    edge_noise = np.random.normal(0, params.edge_noise, img_array.shape).astype(np.float32)
    edge_mask = edges[:, :, np.newaxis] / 255.0
    
    # Apply edge noise with more intensity
    return img_array + (edge_noise * edge_mask * 2)

def add_scanner_effects(image, params):
    """Add various scanner-like effects to an image"""
    # Convert to float32 for better precision
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Add rotation
    height, width = img_array.shape[:2]
    center = (width/2, height/2)
    rotation_matrix = cv2.getRotationMatrix2D(center, params.rotation, 1.0)
    img_array = cv2.warpAffine(img_array, rotation_matrix, (width, height), 
                              borderMode=cv2.BORDER_REPLICATE)
    
    # Initial blur to soften text edges
    img_array = cv2.GaussianBlur(img_array, (3, 3), params.blur_radius)
    
    # Add contrast to make text edges more pronounced
    img_array = np.clip((img_array - 0.5) * params.contrast + 0.5 + params.brightness, 0, 1.0)
    
    # Convert to uint8 for text noise operation
    img_array_uint8 = (img_array * 255).astype(np.uint8)
    
    # Add noise to text
    img_array_uint8 = add_text_noise(img_array_uint8, params)
    
    # Add dust and artifacts (before edge degradation)
    img_array_uint8 = add_dust_and_artifacts(img_array_uint8, params)
    
    # Convert back to float32
    img_array = img_array_uint8.astype(np.float32) / 255.0
    
    # Degrade text edges
    img_array = degrade_text_edges(np.uint8(img_array * 255), params) / 255.0
    
    # Add subtle noise to whole image
    noise = np.random.normal(0, params.noise_level, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 1.0)
    
    # Add light variation
    light_gradient = np.linspace(params.light_variation, 1, height)
    light_gradient = np.tile(light_gradient.reshape(-1, 1), (1, width))
    light_gradient = np.dstack([light_gradient] * 3)
    img_array = np.clip(img_array * light_gradient, 0, 1.0)
    
    # Apply color tint
    img_array = np.clip(img_array * params.color_tint, 0, 1.0)
    
    # Convert back to uint8
    img_array = (img_array * 255).astype(np.uint8)
    
    return Image.fromarray(img_array)

def compress_pdf(input_pdf_path, output_pdf_path):
    """
    Compress PDF by converting to JPEG with optimized quality settings
    and reassembling into a new PDF
    """
    # Convert PDF to images
    print("\nExtracting pages for compression...")
    pages = convert_from_path(input_pdf_path, 200)  # Lower DPI for compression
    
    # Create a temporary directory for processing
    temp_dir = f"temp_compress_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(temp_dir, exist_ok=True)
    compressed_images = []
    
    try:
        # Process each page
        print("Compressing pages:")
        for i, page in enumerate(tqdm(pages)):
            # Convert to grayscale for better compression
            gray_page = page.convert('L')
            
            # Save with aggressive JPEG compression
            temp_path = os.path.join(temp_dir, f'compressed_page_{i}.jpg')
            gray_page.save(
                temp_path, 
                'JPEG', 
                quality=40,  # Aggressive JPEG compression
                optimize=True,  # Use optimal encoder settings
                progressive=True  # Create progressive JPEG
            )
            compressed_images.append(temp_path)
        
        # Combine compressed images into PDF
        print("\nReassembling PDF...")
        with open(output_pdf_path, "wb") as f:
            f.write(img2pdf.convert(
                compressed_images,
                compress=True
            ))
        
        # Print compression statistics
        input_size = os.path.getsize(input_pdf_path)
        output_size = os.path.getsize(output_pdf_path)
        reduction = (1 - (output_size / input_size)) * 100
        
        print(f"\nCompression Results:")
        print(f"├── Original Size: {input_size / 1024 / 1024:.2f} MB")
        print(f"├── Compressed Size: {output_size / 1024 / 1024:.2f} MB")
        print(f"└── Size Reduction: {reduction:.1f}%")
        
    finally:
        # Cleanup
        for img in compressed_images:
            if os.path.exists(img):
                os.remove(img)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
# def compress_pdf(input_pdf_path, output_pdf_path):
#     """Compress PDF file"""
#     reader = PdfReader(input_pdf_path)
#     writer = PdfWriter()

#     for page in reader.pages:
#         page.compress_content_streams()  # This is CPU intensive!
#         writer.add_page(page)

#     writer.add_metadata(reader.metadata)
    
#     with open(output_pdf_path, "wb") as output_file:
#         writer.write(output_file)

def simulate_scan(input_pdf, output_pdf):
    """Main function to simulate scanner effects on a PDF"""
    # Create a temporary directory for image processing
    temp_dir = "temp_scan_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(temp_dir, exist_ok=True)
    
    processed_images = []
    temp_pdf = None
    
    try:
        # Generate single set of parameters for the whole document
        params = ScannerParams()
        params.print_params()
        
        # Convert PDF to images
        print("\nConverting PDF to images...")
        pages = convert_from_path(input_pdf, 300)  # 300 DPI
        
        # Process each page
        print("\nApplying scanner effects:")
        for page in tqdm(pages, desc="Processing pages"):
            # Apply scanner effects
            processed_page = add_scanner_effects(page, params)
            
            # Save processed image temporarily
            temp_image_path = os.path.join(temp_dir, f"page_{len(processed_images)}.jpg")
            processed_page.save(temp_image_path, "JPEG", quality=params.jpeg_quality)
            processed_images.append(temp_image_path)
        
        # Create temporary uncompressed PDF
        temp_pdf = os.path.join(temp_dir, "temp_output.pdf")
        print("\nConverting processed images to PDF...")
        with open(temp_pdf, "wb") as f:
            f.write(img2pdf.convert(processed_images))
        
        # Compress the PDF
        print("\nCompressing PDF...")
        compress_pdf(temp_pdf, output_pdf)
        
        # Print compression results
        if os.path.exists(temp_pdf) and os.path.exists(output_pdf):
            original_size = os.path.getsize(temp_pdf)
            compressed_size = os.path.getsize(output_pdf)
            compression_ratio = (1 - compressed_size / original_size) * 100
            print(f"\nCompression results:")
            print(f"├── Original size: {original_size / 1024 / 1024:.2f} MB")
            print(f"├── Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
            print(f"└── Compression ratio: {compression_ratio:.1f}%")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        raise
        
    finally:
        # Clean up temporary files
        print("\nCleaning up temporary files...")
        for image_path in processed_images:
            if os.path.exists(image_path):
                os.remove(image_path)
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

if __name__ == "__main__":
    input_file = "/Users/tony/Documents/ocr-machine/training_strikethrough/trainingSet/og/STRIKE Partnership Interest Purchase Agreement (Litsey)(298259587.15).pdf"
    for i in range(1):
        output_file = f"trainingSet/true_pdf/scanned_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        # Extract the directory path from the output file
        output_dir = os.path.dirname(output_file)
        
        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directory created: {output_dir}")
        else:
            print(f"Directory already exists: {output_dir}")
        simulate_scan(input_file, output_file)