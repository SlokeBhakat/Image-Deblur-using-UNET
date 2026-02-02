# Image Deblurring Application

## Project Overview
This project implements an end-to-end pipeline for restoring blurred images using a deep learning–based **U-Net architecture** built with **TensorFlow/Keras**.  
The codebase is modular and structured to ensure clarity, maintainability, and easy extension.

---

## Component Breakdown

---

### **1. `config.py`**
**Role:** Centralizes all important configuration parameters and file paths.

**Key Variables**
- **`IMGWIDTH`, `IMGHEIGHT`** – Target resolution (**256×256**) for model input and output.  
- **`SCRIPTDIR`, `ROOTDIR`** – Paths for script and root directories.  
- **`MODELFILENAME`, `MODELPATH`** – Pre-trained U-Net weight file settings.  
- **`INPUTIMAGEPATH`** – Path to the blurred image to be processed.

---

### **2. `utils.py`**
**Role:** Utility functions for image preprocessing and visualization.

**Main Functions**
- **`loadimage(path)`**
  - Reads image from disk using OpenCV.
  - Converts **BGR → RGB**.
  - Resizes to **256×256**.
  - Normalizes pixel values.
  - Ensures standardized input for the model.

- **`showresults(original, deblurred)`**
  - Displays original and deblurred images side-by-side using Matplotlib.
  - Helps visually evaluate model performance.

---

### **3. `unet_archi.py`**
**Role:** Contains the U-Net model architecture and its building blocks.

**Contents**
- **Encoder Block**
  - Two Conv2D layers with BatchNorm + ReLU.
  - MaxPooling layer.
  - **Purpose:** Feature extraction and downsampling.

- **Decoder Block**
  - Upsampling.
  - Skip connection concatenation.
  - Convolution + activation layers.
  - **Purpose:** Reconstruct high-resolution output.

- **`builddeblurringcnn()`**
  - Assembles full U-Net architecture.
  - Implements **residual learning** (adds input to output).
  - Final output clipped to the range **[0, 1]**.

---

### **4. `model_loader.py`**
**Role:** Handles model loading and inference.

**Main Functions**
- **`loadmodel(path)`**
  - Builds U-Net architecture.
  - Loads `.h5` pretrained weights.
  - Includes error handling for missing or corrupted files.

- **`predict(model, img)`**
  - Adds batch dimension.
  - Runs forward inference.
  - Removes batch dimension and clips output.
  - Returns deblurred image.

---

### **5. `main.py`**
**Role:** Main executable script orchestrating the entire deblurring pipeline.

**Workflow**
1. Prints title banner.
2. **Loads input image** using `utils.loadimage`, with path validation.
3. **Loads U-Net model** via `model_loader.loadmodel`.
4. **Runs prediction** to restore the image.
5. **Displays results** with `utils.showresults`.
6. Provides meaningful error messages for missing files or failures.

---

## How to Use

### **1. Update Configuration**
Modify values in `config.py` to customize:
- Image path  
- Model path  
- Directory settings  

### **2. Run the Deblurring Pipeline**
```bash
python main.py
