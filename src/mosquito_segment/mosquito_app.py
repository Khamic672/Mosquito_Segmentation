"""
Mosquito Counter Gradio Application
Frontend interface for mosquito segmentation and counting
Connects to Flask backend API
"""

import gradio as gr
import requests
import base64
from PIL import Image
import io
import numpy as np

# Backend API configuration
API_URL = "http://localhost:5000"

def check_backend_health():
    """Check if backend is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}

def encode_image(image):
    """Encode PIL image to base64"""
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f'data:image/png;base64,{img_str}'

def segment_and_count(image, points_per_side, pred_iou_thresh, stability_score_thresh, min_area):
    """
    Send image to backend for segmentation and counting
    """
    if image is None:
        return None, "Please upload an image first", ""
    
    # Check backend health
    health = check_backend_health()
    if health.get('status') != 'healthy':
        return None, f"Backend not available: {health.get('message', 'Unknown error')}", ""
    
    try:
        # Update configuration first
        config_data = {
            'points_per_side': int(points_per_side),
            'pred_iou_thresh': float(pred_iou_thresh),
            'stability_score_thresh': float(stability_score_thresh),
            'min_mask_region_area': int(min_area)
        }
        
        config_response = requests.post(f"{API_URL}/config", json=config_data, timeout=10)
        
        if config_response.status_code != 200:
            return None, "Failed to update configuration", ""
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Encode image
        image_data = encode_image(image)
        
        # Send to backend
        response = requests.post(
            f"{API_URL}/segment",
            json={'image_data': image_data},
            timeout=60  # Segmentation can take time
        )
        
        if response.status_code != 200:
            return None, f"Error: {response.json().get('error', 'Unknown error')}", ""
        
        result = response.json()
        
        if not result.get('success'):
            return None, "Segmentation failed", ""
        
        # Decode annotated image
        annotated_img_data = result['annotated_image'].split(',')[1]
        annotated_img = Image.open(io.BytesIO(base64.b64decode(annotated_img_data)))
        
        # Format results
        count = result['mosquito_count']
        status_text = f"✅ **Found {count} mosquito{'s' if count != 1 else ''}**"
        
        # Format mask details
        details = "### Detection Details:\n\n"
        for mask in result['masks'][:10]:  # Show first 10
            details += f"**Mosquito #{mask['id']}:**\n"
            details += f"- Area: {mask['area']} pixels\n"
            details += f"- Confidence: {mask['predicted_iou']:.2%}\n"
            details += f"- Stability: {mask['stability_score']:.2%}\n\n"
        
        if len(result['masks']) > 10:
            details += f"*...and {len(result['masks']) - 10} more*"
        
        return annotated_img, status_text, details
    
    except requests.exceptions.Timeout:
        return None, "⏱️ Request timed out. The image might be too large or complex.", ""
    except Exception as e:
        return None, f"❌ Error: {str(e)}", ""

def create_demo():
    """Create Gradio interface"""
    
    with gr.Blocks(title="Mosquito Counter", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🦟 Mosquito Segmentation & Counter
        
        Upload an image of mosquitos and let SAM (Segment Anything Model) detect and count them!
        
        **How to use:**
        1. Upload an image containing mosquitos
        2. Adjust parameters if needed (optional)
        3. Click "Count Mosquitos"
        4. View the annotated image and count results
        """)
        
        # Check backend status
        health = check_backend_health()
        if health.get('status') == 'healthy' and health.get('model_loaded'):
            gr.Markdown("✅ **Backend Status:** Connected and ready")
        else:
            gr.Markdown("⚠️ **Backend Status:** Not available. Please start the backend server first.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                input_image = gr.Image(
                    label="Upload Mosquito Image",
                    type="pil",
                    sources=["upload", "clipboard"]
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    gr.Markdown("Adjust these parameters to fine-tune detection:")
                    
                    points_per_side = gr.Slider(
                        minimum=16,
                        maximum=64,
                        value=32,
                        step=8,
                        label="Points per Side",
                        info="Higher = more detailed segmentation (slower)"
                    )
                    
                    pred_iou_thresh = gr.Slider(
                        minimum=0.5,
                        maximum=0.99,
                        value=0.86,
                        step=0.01,
                        label="Prediction IOU Threshold",
                        info="Higher = stricter detection"
                    )
                    
                    stability_score_thresh = gr.Slider(
                        minimum=0.5,
                        maximum=0.99,
                        value=0.92,
                        step=0.01,
                        label="Stability Score Threshold",
                        info="Higher = more stable masks"
                    )
                    
                    min_area = gr.Slider(
                        minimum=10,
                        maximum=500,
                        value=100,
                        step=10,
                        label="Minimum Mask Area (pixels)",
                        info="Filter out very small detections"
                    )
                
                count_btn = gr.Button("🔍 Count Mosquitos", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                output_image = gr.Image(label="Annotated Image")
                status_text = gr.Markdown()
                details_text = gr.Markdown()
        
        # Examples
        gr.Markdown("### 📝 Tips")
        gr.Markdown("""
        - **Good lighting:** Clear images work best
        - **Contrast:** Mosquitos should be distinguishable from background
        - **Resolution:** Higher resolution captures more details
        - **Adjust parameters:** If count seems off, try adjusting advanced settings
        """)
        
        # Event handler
        count_btn.click(
            fn=segment_and_count,
            inputs=[input_image, points_per_side, pred_iou_thresh, 
                   stability_score_thresh, min_area],
            outputs=[output_image, status_text, details_text]
        )
    
    return demo

if __name__ == "__main__":
    print("Starting Mosquito Counter Gradio App...")
    print(f"Connecting to backend at: {API_URL}")
    print("\nMake sure the backend server is running first!")
    print("Start backend with: python mosquito_backend.py\n")
    
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )