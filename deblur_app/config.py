# deblur_app/config.py

from pathlib import Path

# --- Image Configuration ---
IMG_WIDTH = 256
IMG_HEIGHT = 256

## --- Path Configuration ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

# Model path - CHANGE THIS LINE
MODEL_FILENAME = "unet_deblur.weights.h5"  # Weights file
MODEL_PATH = ROOT_DIR / "models" / MODEL_FILENAME

# Input image path
INPUT_IMAGE_PATH = ROOT_DIR / "samples" / "blurred_image_10.png"