import Quartz.CoreGraphics as CG
import Quartz.ImageIO as ImageIO
from PIL import Image
import tempfile
import os

def quartz_convert_from_path(pdf_path, dpi=300, output_format="RGB"):
    """
    Converts a PDF to a list of PIL images using Quartz CoreGraphics.
    Rotates images to match pdf2image orientation.

    Args:
        pdf_path (str): Path to the PDF file.
        dpi (int): Desired resolution (dots per inch).
        output_format (str): Output image format ("RGB" or "RGBA").

    Returns:
        list: List of PIL.Image objects, one for each page in the PDF.
    """
    # print("IM NEW")
    # Open the PDF document
    pdf_doc = CG.CGPDFDocumentCreateWithURL(CG.CFURLCreateFromFileSystemRepresentation(
        None, bytes(pdf_path, "utf-8"), len(pdf_path), False
    ))
    if not pdf_doc:
        raise ValueError(f"Failed to open PDF: {pdf_path}")

    images = []
    num_pages = CG.CGPDFDocumentGetNumberOfPages(pdf_doc)

    for page_num in range(1, num_pages + 1):  # Pages are 1-indexed in CoreGraphics
        # Get the current page
        pdf_page = CG.CGPDFDocumentGetPage(pdf_doc, page_num)
        media_box = CG.CGPDFPageGetBoxRect(pdf_page, CG.kCGPDFMediaBox)

        # Calculate the target size in pixels
        width_px = int(media_box.size.width * dpi / 72)  # 72 DPI is the base for Quartz
        height_px = int(media_box.size.height * dpi / 72)

        # Create a bitmap context
        color_space = CG.CGColorSpaceCreateDeviceRGB()
        bitmap_context = CG.CGBitmapContextCreate(
            None, width_px, height_px, 8, 4 * width_px, color_space, CG.kCGImageAlphaPremultipliedLast
        )

        if not bitmap_context:
            raise RuntimeError("Failed to create bitmap context.")

        # Scale and draw the page into the context
        CG.CGContextScaleCTM(bitmap_context, dpi / 72, dpi / 72)  # Scale context to match DPI
        CG.CGContextDrawPDFPage(bitmap_context, pdf_page)

        # Create a CGImage from the context
        quartz_image = CG.CGBitmapContextCreateImage(bitmap_context)

        # Save the image temporarily and load it with PIL
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            dest = ImageIO.CGImageDestinationCreateWithURL(
                CG.CFURLCreateFromFileSystemRepresentation(None, bytes(temp_file_path, "utf-8"), len(temp_file_path), False),
                "public.png", 1, None  # Use the public UTI identifier for PNG
            )
            ImageIO.CGImageDestinationAddImage(dest, quartz_image, None)
            ImageIO.CGImageDestinationFinalize(dest)

            # Open the image with PIL and rotate it to match pdf2image orientation
            image = Image.open(temp_file_path).convert(output_format)
            # Rotate 90 degrees clockwise to match pdf2image orientation
            image = image.rotate(-90, expand=True)
            images.append(image)

            # Clean up the temporary file
            os.unlink(temp_file_path)

    return images