import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import os

def load_image(image_path):
    """Loads an image and converts it to a NumPy array, handling RGB."""
    img = Image.open(image_path).convert('RGB') # Convert to RGB
    return np.array(img) / 255.0 # Normalize to [0, 1]

def extract_patches(image, patch_size, stride):
    """Extracts overlapping patches from an image (supports RGB)."""
    patches = []
    positions = [] # Store the (i, j) top-left corner of each patch
    h, w, c = image.shape # Now includes channels
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size, :] # Extract all channels
            patches.append(patch)
            positions.append((i, j))
    return np.array(patches), positions # Return positions as well

def create_damaged_patch(patch, damage_size_ratio_min=0.1, damage_size_ratio_max=0.4):
    """
    Creates a damaged version of a patch by adding a random black square (for RGB).
    """
    damaged_patch = np.copy(patch)
    damage_mask = np.zeros_like(damaged_patch, dtype=bool) # Use bool for mask
    p_h, p_w, p_c = patch.shape # Now includes channels

    # Determine random damage size
    damage_h_min = int(p_h * damage_size_ratio_min)
    damage_h_max = int(p_h * damage_size_ratio_max)
    damage_w_min = int(p_w * damage_size_ratio_min)
    damage_w_max = int(p_w * damage_size_ratio_max)

    damage_h = np.random.randint(damage_h_min, damage_h_max + 1)
    damage_w = np.random.randint(damage_w_min, damage_w_max + 1)

    # Determine random position for the damage
    start_row = np.random.randint(0, p_h - damage_h + 1)
    start_col = np.random.randint(0, p_w - damage_w + 1)

    # Set the damaged area to black across all channels
    damaged_patch[start_row : start_row + damage_h,
                  start_col : start_col + damage_w, :] = 0.0 # Set all channels to 0.0 (black)

    damage_mask[start_row : start_row + damage_h,
                start_col : start_col + damage_w, :] = True

    return damage_mask, damaged_patch

def build_autoencoder(input_shape):
    """Builds a simple convolutional autoencoder (supports RGB input/output)."""
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoder_input)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    # The last layer's filter count must match the number of channels (3 for RGB)
    decoded = layers.Conv2D(input_shape[-1], (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = models.Model(encoder_input, decoded)
    return autoencoder

def inpaint_image_with_sliding_window(damaged_image, damage_mask, trained_model, patch_size, stride):
    """
    Inpaints a damaged image by sliding a window, predicting missing parts with a
    trained model, and reconstructing the image.

    Args:
        damaged_image (np.ndarray): The input image with damaged regions (normalized).
        trained_model (tf.keras.Model): The trained inpainting autoencoder model.
        patch_size (int): The size of the square patches (e.g., 64).
        stride (int): The step size for sliding the window.

    Returns:
        np.ndarray: The reconstructed (inpainted) image.
    """
    h, w, c = damaged_image.shape
    reconstructed_image = np.copy(damaged_image)
    overlap_count = np.zeros(damaged_image.shape) # To handle averaging overlapping regions

    # Extract patches and their positions from the *damaged* image
    damaged_patches, positions = extract_patches(damaged_image, patch_size, stride)

    print(f"Inpainting image of size {h}x{w} with {len(damaged_patches)} patches...")

    # Predict all patches at once for efficiency
    inpainted_patches = trained_model.predict(damaged_patches, verbose=0)

    for idx, (i, j) in enumerate(positions):
        # The predicted patch directly gives the inpainted content
        predicted_patch = inpainted_patches[idx]
        patch_damage_mask = damage_mask[i:i+patch_size, j:j+patch_size]
        multi_channel_mask_for_patch = np.stack([patch_damage_mask]*3, axis=-1)

        if patch_damage_mask.any():
            # Add the predicted patch to the reconstructed image
            reconstructed_image[i:i+patch_size, j:j+patch_size, :][multi_channel_mask_for_patch] += predicted_patch[multi_channel_mask_for_patch]
            overlap_count[i:i+patch_size, j:j+patch_size, :] += 1

    # Avoid division by zero for areas that weren't covered by any patch
    overlap_count[overlap_count == 0] = 1

    # Average the overlapping regions
    final_inpainted_image = reconstructed_image / overlap_count

    # Clip values to ensure they are within [0, 1] range after averaging
    final_inpainted_image = np.clip(final_inpainted_image, 0, 1)

    return final_inpainted_image


# --- Main part of the script ---
if __name__ == "__main__":
    # 1. Load the image
    image_path = os.path.join(os.path.dirname(__file__), '..', 'TV-Inpainting', 'data', 'Lola.jpg')
    try:
        original_image = load_image(image_path)
        print(f"Loaded image from {image_path} with shape {original_image.shape}")
    except FileNotFoundError:
        print("Sample RGB image not found. Creating a dummy RGB image for demonstration.")
        # Create a 256x256 RGB random image
        original_image = np.random.rand(256, 256, 3)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        Image.fromarray((original_image * 255).astype(np.uint8)).save(image_path)
        original_image = load_image(image_path)


    # 2. Extract patches for training
    patch_size = 64
    stride = 32 # Overlapping patches
    all_patches, _ = extract_patches(original_image, patch_size, stride) # _ to ignore positions for training
    print(f"Extracted {len(all_patches)} patches, each of size {patch_size}x{patch_size}.")

    # Input shape for CNN (batch, height, width, channels)
    input_shape = all_patches.shape[1:]

    # 3. Create damaged patches and prepare dataset
    damaged_patches = np.array([patch for (mask, patch) in [create_damaged_patch(p) for p in all_patches]])

    print(f"Shape of original patches dataset: {all_patches.shape}")
    print(f"Shape of damaged patches dataset: {damaged_patches.shape}")

    # 4. Build the autoencoder
    def SSIMLoss(y_true, y_pred):
        return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

    autoencoder = build_autoencoder(input_shape)
    autoencoder.compile(optimizer='adam', loss='mae') # Using MAE for simplicity as requested, but SSIMLoss is available
    autoencoder.summary()

    # 5. Train the autoencoder
    print("\nTraining the autoencoder...")
    history = autoencoder.fit(damaged_patches, all_patches,
                              epochs=50,
                              batch_size=32,
                              shuffle=True,
                              validation_split=0.1,
                              verbose=1)

    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error') # Updated label based on 'mae' loss
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6. Evaluate the trained model on 3 different (random) artificially damaged patches
    print("\nEvaluating model performance on new damaged patches (individual patches)...")
    num_eval_samples = 3

    random_indices = np.random.choice(len(all_patches), num_eval_samples, replace=False)

    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(random_indices):
        original_patch = all_patches[idx]
        test_damage_mask, test_damaged_patch = create_damaged_patch(original_patch)
        test_damaged_patch_input = np.expand_dims(test_damaged_patch, axis=0)

        predicted_patch = autoencoder.predict(test_damaged_patch_input, verbose=0).squeeze()

        # For individual patch display, we typically show the damaged patch
        # and then the model's output for the *entire* patch
        # or just the inpainted region on the damaged patch.
        # Let's show the model's direct output as 'reconstructed'
        # For a true 'reconstructed' patch, we'd blend or replace damaged areas.
        # Here, 'predicted_patch' is the model's best attempt at the full patch.

        plt.subplot(num_eval_samples, 3, i * 3 + 1)
        plt.imshow(original_patch)
        plt.title('Original Patch')
        plt.axis('off')

        plt.subplot(num_eval_samples, 3, i * 3 + 2)
        plt.imshow(test_damaged_patch)
        plt.title('Damaged Patch')
        plt.axis('off')

        plt.subplot(num_eval_samples, 3, i * 3 + 3)
        plt.imshow(predicted_patch) # Show the model's direct output for the patch
        plt.title('Model Predicted Patch')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # --- New part: Inpaint the entire image using the sliding window ---
    print("\n--- Inpainting the entire damaged image with sliding window ---")

    # Create a single large damaged image for inpainting
    # For simplicity, let's create a central black square for demonstration
    image_h, image_w, _ = original_image.shape
    damaged_full_image = np.copy(original_image)
    
    damage_start_row = image_h // 4
    damage_end_row = 3 * image_h // 4
    damage_start_col = image_w // 4
    damage_end_col = 3 * image_w // 4

    damaged_full_image[damage_start_row:damage_end_row,
                       damage_start_col:damage_end_col, :] = 0.0
    damage_mask = np.zeros((image_h, image_w))
    damage_mask[damage_start_row:damage_end_row,
                       damage_start_col:damage_end_col] = 1
    damage_mask = damage_mask.astype(bool)

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title('Original Full Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(damaged_full_image)
    plt.title('Damaged Full Image')
    plt.axis('off')

    # Perform the inpainting
    inpainted_full_image = inpaint_image_with_sliding_window(
        damaged_full_image, damage_mask, autoencoder, patch_size, stride
    )

    plt.subplot(1, 3, 3)
    plt.imshow(inpainted_full_image)
    plt.title('Inpainted Full Image (Sliding Window)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nFull image inpainting demonstration complete.")