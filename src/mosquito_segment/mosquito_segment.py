"""
Mosquito Segmentation Backend API using SAM (Segment Anything Model)
Flask API server for mosquito detection and counting
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app)

# Global variables for model
sam_model = None
mask_generator = None

def initialize_sam():
    """Initialize SAM model"""
    global sam_model, mask_generator
    
    # Download SAM checkpoint if not exists
    checkpoint_path = "sam_vit_h_4b8939.pth"
    
    if not os.path.exists(checkpoint_path):
        print("SAM checkpoint not found. Please download from:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        return False
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint_path)
        sam_model.to(device=device)
        
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Minimum area for a mask
        )
        
        print(f"SAM model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"Error loading SAM model: {e}")
        return False

def decode_image(image_data):
    """Decode base64 image or file upload"""
    if isinstance(image_data, str):
        # Base64 encoded image
        image_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
        image = Image.open(BytesIO(image_bytes))
    else:
        # File upload
        image = Image.open(image_data)
    
    return np.array(image.convert('RGB'))

def filter_mosquito_masks(masks, image_shape):
    """
    Filter masks to identify likely mosquito objects
    Based on size, aspect ratio, and shape characteristics
    """
    filtered_masks = []
    h, w = image_shape[:2]
    image_area = h * w
    
    for mask_data in masks:
        mask = mask_data['segmentation']
        area = mask_data['area']
        bbox = mask_data['bbox']  # x, y, w, h
        
        # Filter by relative size (mosquitos are typically small objects)
        relative_area = area / image_area
        if relative_area < 0.0001 or relative_area > 0.3:
            continue
        
        # Filter by aspect ratio (mosquitos are elongated)
        aspect_ratio = bbox[2] / (bbox[3] + 1e-6)
        if aspect_ratio < 0.2 or aspect_ratio > 5.0:
            continue
        
        # Filter by minimum absolute size
        if area < 50:
            continue
        
        filtered_masks.append(mask_data)
    
    return filtered_masks

def create_visualization(image, masks):
    """Create visualization with masks overlaid"""
    overlay = image.copy()
    
    # Create color map for different masks
    np.random.seed(42)
    
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation']
        color = np.random.randint(0, 255, 3).tolist()
        
        # Apply colored mask
        overlay[mask] = overlay[mask] * 0.5 + np.array(color) * 0.5
        
        # Draw bounding box
        bbox = mask_data['bbox']
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        
        # Add number label
        cv2.putText(overlay, str(i + 1), (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return overlay

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': mask_generator is not None
    })

@app.route('/segment', methods=['POST'])
def segment_mosquitos():
    """
    Main endpoint for mosquito segmentation
    Expects: image file or base64 encoded image
    Returns: count and annotated image
    """
    if mask_generator is None:
        return jsonify({'error': 'Model not initialized'}), 500
    
    try:
        # Get image from request
        if 'image' in request.files:
            image = decode_image(request.files['image'])
        elif 'image_data' in request.json:
            image = decode_image(request.json['image_data'])
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Generate masks
        print("Generating masks...")
        masks = mask_generator.generate(image)
        print(f"Generated {len(masks)} total masks")
        
        # Filter for mosquito-like objects
        mosquito_masks = filter_mosquito_masks(masks, image.shape)
        mosquito_count = len(mosquito_masks)
        print(f"Filtered to {mosquito_count} mosquito candidates")
        
        # Create visualization
        annotated_image = create_visualization(image, mosquito_masks)
        
        # Convert to base64 for response
        pil_image = Image.fromarray(annotated_image.astype('uint8'))
        buffer = BytesIO()
        pil_image.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Prepare mask details
        mask_details = []
        for i, mask_data in enumerate(mosquito_masks):
            mask_details.append({
                'id': i + 1,
                'area': int(mask_data['area']),
                'bbox': [int(v) for v in mask_data['bbox']],
                'predicted_iou': float(mask_data['predicted_iou']),
                'stability_score': float(mask_data['stability_score'])
            })
        
        return jsonify({
            'success': True,
            'mosquito_count': mosquito_count,
            'annotated_image': f'data:image/png;base64,{img_str}',
            'masks': mask_details
        })
    
    except Exception as e:
        print(f"Error during segmentation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/config', methods=['POST'])
def update_config():
    """Update segmentation parameters"""
    global mask_generator
    
    try:
        config = request.json
        
        # Reinitialize mask generator with new parameters
        mask_generator = SamAutomaticMaskGenerator(
            model=sam_model,
            points_per_side=config.get('points_per_side', 32),
            pred_iou_thresh=config.get('pred_iou_thresh', 0.86),
            stability_score_thresh=config.get('stability_score_thresh', 0.92),
            min_mask_region_area=config.get('min_mask_region_area', 100),
        )
        
        return jsonify({'success': True, 'message': 'Configuration updated'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Initializing SAM model...")
    if initialize_sam():
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Failed to initialize SAM model. Please ensure checkpoint is downloaded.")
        print("\nTo download SAM checkpoint:")
        print("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")