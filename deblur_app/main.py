# deblur_app/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from config import MODEL_PATH, INPUT_IMAGE_PATH, IMG_WIDTH, IMG_HEIGHT
from model_loader import load_model, predict
from utils import load_image, show_results

def main():
    print("="*60)
    print("IMAGE DEBLURRING")
    print("="*60)
    
    # Check image exists
    print(f"\n1. Loading image: {INPUT_IMAGE_PATH.name}")
    if not INPUT_IMAGE_PATH.exists():
        print(f"   ERROR: Image not found at {INPUT_IMAGE_PATH}")
        print(f"   Please edit INPUT_IMAGE_PATH in config.py")
        return
    
    original = load_image(INPUT_IMAGE_PATH)
    print(f"Loaded and resized to {IMG_WIDTH}x{IMG_HEIGHT}")
    
    
    # Load model
    print(f"\n2. Loading model: {MODEL_PATH.name}")
    if not MODEL_PATH.exists():
        print(f"   ERROR: Model not found at {MODEL_PATH}")
        return
    
    model = load_model(MODEL_PATH)
    if model is None:
        return
    print(f"Model loaded")
    
    # Deblur
    print(f"\n4. Deblurring...")
    deblurred = predict(model, original)
    print(f"Done")
    
    # Show results
    print(f"\n5. Showing results...")
    show_results(original, deblurred)
    print(f"\n✓ Complete!")

if __name__ == "__main__":
    main()