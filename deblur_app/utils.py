# deblur_app/utils.py

import matplotlib
matplotlib.use('TkAgg')

import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import IMG_WIDTH, IMG_HEIGHT

def load_image(path):
    """
    Load image and resize to (128, 128, 3) with values [0, 1].
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    return img.astype(np.float32) / 255.0

def show_results(original, deblurred):
    """Display the three images side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title("Original (Blurred)")
    axes[0].axis('off')

    
    axes[1].imshow(deblurred)
    axes[1].set_title("Deblurred")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()