from PIL import Image, ImageDraw
import sys
from pathlib import Path

def draw_bbox(image_path, coords, output_path=None):
    """
    Draw a bounding box on an image and save or display it.
    
    Args:
        image_path (str): Path to the input image
        coords (tuple): Coordinates (x1, y1, x2, y2) for the bounding box
        output_path (str, optional): Path to save the output image. If None, displays the image.
    """
    # Open the image
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    # Create a copy to draw on
    img_with_box = img.copy()
    draw = ImageDraw.Draw(img_with_box)
    
    # Draw the bounding box
    # Using red color with 50% opacity
    box_color = (255, 0, 0, 127)  # RGBA
    # Draw filled rectangle with transparency
    draw.rectangle(coords, outline=(255, 0, 0), width=3)  # Solid red outline
    
    # Create a transparent overlay for the fill
    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    draw_overlay.rectangle(coords, fill=box_color)
    
    # Convert the original image to RGBA if it isn't already
    if img_with_box.mode != 'RGBA':
        img_with_box = img_with_box.convert('RGBA')
    
    # Combine the images
    img_with_box = Image.alpha_composite(img_with_box, overlay)
    
    # Add coordinate text
    text_color = (255, 255, 255)  # White
    text_outline = (0, 0, 0)      # Black outline
    x1, y1, x2, y2 = coords
    text = f"({int(x1)},{int(y1)})"
    # Add text at top-left corner
    draw = ImageDraw.Draw(img_with_box)
    draw.text((x1, y1-20), text, fill=text_color, stroke_width=2, stroke_fill=text_outline)
    # Add text at bottom-right corner
    text = f"({int(x2)},{int(y2)})"
    draw.text((x2-100, y2+5), text, fill=text_color, stroke_width=2, stroke_fill=text_outline)
    
    if output_path:
        try:
            img_with_box.save(output_path)
            print(f"Image saved to {output_path}")
        except Exception as e:
            print(f"Error saving image: {e}")
    else:
        img_with_box.show()

def adjust_bounding_box(original_image_size, gpt_image_size, gpt_table_coords):
    """
    Adjusts GPT bounding boxes to match the original image dimensions.

    Parameters:
        original_image_size (tuple): (original_width, original_height)
        gpt_image_size (tuple): (gpt_width, gpt_height)
        gpt_table_coords (tuple): (left, top, right, bottom)

    Returns:
        tuple: Adjusted bounding box coordinates (left, top, right, bottom)
    """
    original_width, original_height = original_image_size
    gpt_width, gpt_height = gpt_image_size
    left, top, right, bottom = gpt_table_coords

    scale_x = original_width / gpt_width
    scale_y = original_height / gpt_height

    adjusted_left = left * scale_x
    adjusted_top = top * scale_y
    adjusted_right = right * scale_x
    adjusted_bottom = bottom * scale_y

    return (adjusted_left, adjusted_top, adjusted_right, adjusted_bottom)

def main():
    # GPT-reported image sizes and table coordinates for each page
    gpt_data_by_page = {
        1: {
            'gpt_image_size': (1275, 1650),
            'gpt_table_coords': (145, 430, 1130, 1070)
        },
        2: {
            'gpt_image_size': (2550, 3300),
            'gpt_table_coords': (144, 276, 2406, 2760)
        }
    }
    
    # Base directory where your PDF images are stored
    base_dir = Path(__file__).parent.parent / 'images'
    output_dir = Path("bbox_output")
    output_dir.mkdir(exist_ok=True)
    
    # Process each page
    for page_num, data in gpt_data_by_page.items():
        # Construct input and output paths
        input_path = base_dir / f"page_{page_num}.png"  # Adjust filename pattern as needed
        output_path = output_dir / f"bbox_page_{page_num}.png"
        
        if input_path.exists():
            print(f"Processing page {page_num}...")
            # Open the original image to get its size
            img = Image.open(input_path)
            original_image_size = img.size  # (width, height)
            img.close()
            
            # Get GPT data
            gpt_image_size = data['gpt_image_size']
            gpt_table_coords = data['gpt_table_coords']
            
            # Adjust bounding box coordinates
            adjusted_coords = adjust_bounding_box(original_image_size, gpt_image_size, gpt_table_coords)
            
            # Draw bounding box on image
            draw_bbox(str(input_path), adjusted_coords, str(output_path))
        else:
            print(f"Warning: Image file not found for page {page_num}: {input_path}")

if __name__ == "__main__":
    main()
