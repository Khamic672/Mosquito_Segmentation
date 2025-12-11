import cv2
import numpy as np
import gradio as gr
from PIL import Image

def count_mosquitoes(image):
    """
    Count mosquitoes in an image using OpenCV blob detection.
    
    Args:
        image: PIL Image or numpy array
        
    Returns:
        tuple: (annotated_image, count, description)
    """
    if image is None:
        return None, 0, "No image provided"
    
    # Convert PIL Image to numpy array if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
    else:
        img = image
    
    # Convert to BGR if RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area (adjust these values based on your images)
    min_area = 50  # Minimum area for a mosquito
    max_area = 5000  # Maximum area for a mosquito
    
    mosquito_count = 0
    result_img = img_bgr.copy()
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            # Draw bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            mosquito_count += 1
            
            # Draw contour
            cv2.drawContours(result_img, [contour], -1, (255, 0, 0), 1)
    
    # Convert back to RGB for display
    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    description = f"Detected {mosquito_count} mosquito(es) in the image."
    
    return result_rgb, mosquito_count, description

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Mosquito Counter") as app:
        gr.Markdown("# 🦟 Mosquito Counter")
        gr.Markdown("Upload or drag & drop an image to count mosquitoes")
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
                submit_btn = gr.Button("Count Mosquitoes", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Detected Mosquitoes")
                count_output = gr.Number(label="Mosquito Count")
                description_output = gr.Textbox(label="Details")
        
        # Set up the event
        submit_btn.click(
            fn=count_mosquitoes,
            inputs=input_image,
            outputs=[output_image, count_output, description_output]
        )
        
        # Also allow automatic counting on image upload
        input_image.change(
            fn=count_mosquitoes,
            inputs=input_image,
            outputs=[output_image, count_output, description_output]
        )
    
    return app

if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False)  # Set share=True if you want a public link