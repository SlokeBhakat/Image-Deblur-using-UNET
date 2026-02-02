# deblur_app/model_loader.py

from tensorflow import keras
import numpy as np
from unet_archi import build_deblurring_cnn

def load_model(path):
    try:
        print(f"Building U-Net deblurring architecture...")
        model = build_deblurring_cnn(input_shape=(256, 256, 3))
        
        print(f"Loading weights from {path.name}...")
        model.load_weights(str(path))
        
        print(f"Model loaded successfully")
        return model
        
    except Exception as e:
        print(f"   ERROR: {e}")
        return None

def predict(model, img):
    """
    """
    batch = np.expand_dims(img, axis=0)
    output = model.predict(batch, verbose=0)
    result = np.squeeze(output, axis=0)
    return np.clip(result, 0, 1)
