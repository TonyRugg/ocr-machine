from PIL import Image
import os
from pathlib import Path
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def analyze_image_dimensions(true_dir, false_dir):
    stats = {
        'widths': [],
        'heights': [],
        'ratios': [],  # width/height ratios
        'pixel_counts': []  # total pixels per image
    }
    
    # Process both directories
    for directory in [true_dir, false_dir]:
        # Using rglob instead of glob for recursive search
        for img_path in tqdm(Path(directory).rglob('*.jpeg')):
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    stats['widths'].append(width)
                    stats['heights'].append(height)
                    stats['ratios'].append(width/height)
                    stats['pixel_counts'].append(width * height)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    # Calculate statistics
    for key in stats:
        data = np.array(stats[key])
        print(f"\n{key.upper()} Statistics:")
        print(f"Min: {data.min():.2f}")
        print(f"Max: {data.max():.2f}")
        print(f"Mean: {data.mean():.2f}")
        print(f"Median: {np.median(data):.2f}")
        print(f"Std Dev: {data.std():.2f}")
        
        # Calculate percentiles
        percentiles = [5, 25, 75, 95]
        for p in percentiles:
            print(f"{p}th percentile: {np.percentile(data, p):.2f}")

    # Print total number of images analyzed
    print(f"\nTotal images analyzed: {len(stats['widths'])}")

if __name__ == "__main__":
    # Use the function
    analyze_image_dimensions("trainingSet/false_images", "trainingSet/true_images")









__doc__ = """
Image Dimension Analysis Tool

This script provides a utility function to analyze the dimensions of images in two directories.
It calculates statistics such as minimum, maximum, mean, median, standard deviation, and percentiles 
for image widths, heights, aspect ratios, and pixel counts.

Dependencies:
    - PIL (from Pillow): For opening and processing images.
    - NumPy: For numerical operations.
    - tqdm: For displaying progress bars.
    - pathlib: For handling filesystem paths.

Function:
    analyze_image_dimensions(true_dir, false_dir):
        Analyzes the dimensions of images in the specified directories and prints statistical summaries.

        Parameters:
            - true_dir (str): Path to the directory containing the first set of images.
            - false_dir (str): Path to the directory containing the second set of images.

        Image Processing:
            - Reads `.jpeg` files recursively from both directories.
            - Extracts width, height, aspect ratio (width/height), and pixel count for each image.

        Output:
            - Prints statistics for each metric:
                - Minimum, maximum, mean, median, and standard deviation.
                - Percentiles: 5th, 25th, 75th, and 95th.
            - Displays the total number of images analyzed.

        Notes:
            - Images that cannot be processed (e.g., corrupted or unsupported formats) are skipped, 
              with a warning printed to the console.

Usage:
    Call the `analyze_image_dimensions` function with paths to two directories of images:
    ```
    analyze_image_dimensions("path/to/true_images", "path/to/false_images")
    ```

    Example Output:
        WIDTHS Statistics:
        Min: 500.00
        Max: 1920.00
        Mean: 1080.50
        Median: 1024.00
        Std Dev: 200.35
        5th percentile: 512.00
        25th percentile: 768.00
        75th percentile: 1440.00
        95th percentile: 1800.00

        Total images analyzed: 150
"""
