import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def add_degradations(image):
    """
    Apply multiple degradations to simulate bad scans:
    - Blurriness
    - Text lightening (more pronounced)
    - Enhanced noise
    
    Ensures at least one degradation is applied every time.
    
    :param image: Input image
    :return: Degraded image, applied effects
    """
    img_float = np.float32(image)
    height, width = img_float.shape[:2]

    applied_effects = []

    # Apply random blur (simulate optical blurriness)
    if np.random.rand() < 0.5:
        kernel_size = np.random.choice([3, 5, 7])  # Small to moderate blur
        img_float = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), 0)
        applied_effects.append(f"blur_kernel{kernel_size}")

    # Apply text lightening (reduce contrast adaptively, stronger effect)
    if np.random.rand() < 0.5 or not applied_effects:  # Ensure at least one effect
        lightening_mask = np.random.uniform(0.1, 0.2, (height, width))  # More aggressive lightening
        lightening_mask = cv2.GaussianBlur(lightening_mask, (15, 15), 0)  # Smooth transitions
        lightening_mask = np.expand_dims(lightening_mask, axis=-1)  # Match channels

        # Only apply lightening to darker regions (threshold bright areas)
        text_mask = (img_float < 200).astype(np.float32)  # Threshold for darker areas
        img_float = img_float * (1 - text_mask + text_mask * lightening_mask)  # Preserve bright areas

        applied_effects.append("lightening")

    # Add enhanced noise for texture
    if np.random.rand() < 0.5 or not applied_effects:  # Ensure at least one effect
        noise = np.random.randn(*img_float.shape) * 10  # Increased Gaussian noise
        img_float += noise
        applied_effects.append("noise")

    # Clip to valid range and convert back to uint8
    degraded_img = np.clip(img_float, 0, 255).astype(np.uint8)

    return degraded_img, applied_effects


def simulate_scans(src_folder, dest_folder, num_images):
    """
    Simulate scans by applying noise to each image in src_folder and saving the results to dest_folder.
    Traverses all subfolders for images.
    
    :param src_folder: Path to the source folder containing images
    :param dest_folder: Path to the destination folder where processed images will be saved
    :param num_images: Number of images to process
    """
    if not os.path.exists(src_folder):
        print(f"Error: Source folder '{src_folder}' does not exist.")
        return

    os.makedirs(dest_folder, exist_ok=True)

    print(f'Source: {src_folder}')
    print(f'Destination: {dest_folder}')
    print(f'Max Images: {num_images}')

    image_count = 0
    for root, _, files in tqdm(os.walk(src_folder), desc=f'Processing Images'):  # Traverse subfolders
        for filename in files:
            if image_count >= num_images:
                break

            src_path = os.path.join(root, filename)
            dest_path = os.path.join(dest_folder, filename)

            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            image = cv2.imread(src_path)
            if image is None:
                print(f"Could not read image {src_path}, skipping.")
                continue

            degraded_image, effects = add_degradations(image)

            # Save the processed image with effects in filename for debugging
            base, ext = os.path.splitext(filename)
            effects_str = "_".join(effects)
            dest_path = os.path.join(dest_folder, f"{base}_{effects_str}{ext}")

            cv2.imwrite(dest_path, degraded_image)
            # print(f"Processed '{filename}' -> '{dest_path}' with effects: {effects_str}")
            image_count += 1

            if image_count >= num_images:
                break
    print(f'Total Images Converted: {image_count}')

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Simulate bad scans by applying degradations to each image in a folder.')
    parser.add_argument(
        '--src', 
        default='trainingSet/false_images/dallas-tx-3_chunk_2/', 
        help='Path to the source folder containing images'
        )
    parser.add_argument(
        '--dest', 
        default='trainingSet/scans/false/SCAN-dallas-tx-3_chunk_2', 
        help='Path to the destination folder where processed images will be saved (default: test_output)'
        )
    parser.add_argument(
        '-n', 
        '--num-images', 
        type=int, 
        default=10, 
        help='Number of images to process (default: 10)')
    args = parser.parse_args()

    # Use provided arguments or defaults
    src_folder = args.src
    dest_folder = args.dest
    num_images = args.num_images

    simulate_scans(src_folder, dest_folder, num_images)

if __name__ == "__main__":
    main()
