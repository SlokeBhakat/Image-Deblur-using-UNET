# deblur_app/unet_architecture.py

import tensorflow as tf
from keras import layers, models

def encoder_block(input_tensor, num_filters):
    """Creates an encoder block (Conv -> BN -> ReLU -> Conv -> ReLU)."""
    # First 3x3 Convolutional layer: extracts features, 'relu' activation for non-linearity.
    # 'padding='same'' ensures the output feature map size is the same as the input.
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(input_tensor)
    
    # Batch Normalization: standardizes the outputs of the previous layer,
    # which stabilizes and speeds up training.
    c = layers.BatchNormalization()(c)
    
    # Second 3x3 Convolutional layer: further processes features.
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(c)
    
    # Max Pooling: reduces the spatial dimensions (H/W) by half (2x2 pool size), 
    # capturing the most important features and increasing the receptive field.
    p = layers.MaxPooling2D((2, 2))(c)
    
    # Return:
    # c: The final convolutional output (used as the skip connection).
    # p: The downsampled (pooled) output (passed to the next encoder block).
    return c, p 

def decoder_block(input_tensor, skip_tensor, num_filters):
    """Creates a decoder block (UpSample -> Concat -> Conv -> ReLU -> Conv -> ReLU)."""
    
    # UpSampling: doubles the spatial dimensions (H/W) of the feature map, 
    # effectively reversing the effect of MaxPooling in the encoder.
    u = layers.UpSampling2D((2, 2))(input_tensor)
    
    # Concatenation (Skip Connection): merges the upsampled feature map (u)
    # with the corresponding feature map (skip_tensor) from the encoder.
    # This transfers fine-grained, high-resolution details lost during pooling.
    u = layers.Concatenate()([u, skip_tensor])
    
    # First 3x3 Convolutional layer: processes the combined feature map.
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(u)
    
    # Second 3x3 Convolutional layer: final processing for the block.
    c = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(c)
    
    # Return: The output feature map of the decoder block.
    return c

def build_deblurring_cnn(input_shape=(256, 256, 3)):
    """
    Build your custom U-Net deblurring architecture.
    """
    # Define the input layer with the expected image shape (e.g., 256x256x3 for RGB).
    inputs = layers.Input(shape=input_shape)

    # --- Encoder (Downsampling Path) ---
    # The image size is progressively halved (256 -> 128 -> 64) and feature maps increase (64 -> 128).
    c1, p1 = encoder_block(inputs, 64) # Output size: 128x128
    c2, p2 = encoder_block(p1, 128) # Output size: 64x64

    # --- Bottleneck (Lowest Resolution) ---
    # Connects the encoder and decoder paths.
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    b = layers.Conv2D(256, 3, activation='relu', padding='same')(b) # Output size: 64x64

    # --- Decoder (Upsampling Path) ---
    # The feature map size is progressively doubled (64 -> 128 -> 256) and features decrease.
    d1 = decoder_block(b, c2, 128) # Uses skip connection c2. Output size: 128x128
    d2 = decoder_block(d1, c1, 64) # Uses skip connection c1. Output size: 256x256

    # --- Output Layer ---
    # 1x1 Convolution: Maps the final 64-feature map to 3 channels (for RGB image output).
    # 'linear' activation is used because the output pixels should not be restricted to 
    # [0, 1] until the final clipping/scaling (or it could be 'sigmoid' if expecting [0,1]).
    outputs = layers.Conv2D(3, 1, activation='linear', padding='same')(d2)

    # --- Residual Section (Key to Deblurring) ---
    # The output of the U-Net path (outputs) is added to the original input image (inputs).
    # This forces the U-Net to learn the *difference* (the blur and noise to be removed) 
    # rather than the entire deblurred image from scratch. 
    # This concept is known as a **Residual Connection** or **Residual Learning**.
    outputs = layers.Add()([outputs, inputs]) 
    
    # Clip the pixel values to ensure they are within the standard range [0.0, 1.0].
    # This is necessary because the 'linear' output activation and the residual addition 
    # might produce values outside the valid range.
    outputs = layers.Lambda(lambda x: tf.clip_by_value(x, 0.0, 1.0))(outputs)

    # Define the model, connecting the input and the final processed output.
    model = models.Model(inputs, outputs, name="U-Net_DeblurringCNN")
    
    return model