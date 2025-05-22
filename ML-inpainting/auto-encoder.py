import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2 # For image loading and manipulation
import time

# --- Configuration ---
PATCH_SIZE = 64  # Size of the square patches (e.g., 64x64 pixels)
MASK_SIZE = 32   # Size of the square mask to apply within the patch
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.005

corruptions = [[(50, 50), (82, 82)],
 [(150, 180), (182, 212)],
 [(50, 50), (82, 82)],
 [(200, 300), (232, 332)],
 [(400, 100), (432, 132)],
 [(600, 500), (632, 532)],
 [(100, 800), (132, 832)]]

# --- 1. Load your Damaged Image ---
# Replace 'your_damaged_image.jpg' with the actual path
try:
    damaged_image_path = os.path.join(os.path.dirname(__file__), '..', 'TV-Inpainting', 'data', 'Lola.jpg')
    # For demonstration, let's create a dummy damaged image if not provided
    try:
        damaged_image_np = cv2.imread(damaged_image_path)
        damage_mask = np.zeros(shape=(damaged_image_np.shape[0], damaged_image_np.shape[1]))
        for ((x_start, y_start), (x_stop, y_stop)) in corruptions:
            cv2.rectangle(damaged_image_np, (x_start, y_start), (x_stop, y_stop), (0, 0, 0), -1)
            damage_mask[y_start : y_stop + 1, x_start : x_stop + 1] = 1
            # print("hello!!")
            # print(damage_mask[:10, :])
            # print(np.any(damage_mask[:10, :10]))
            # exit()
        if damaged_image_np is None:
            raise FileNotFoundError # Handle if image path is invalid
    except FileNotFoundError:
        print(f"'{damaged_image_path}' not found. Creating a dummy damaged image for demonstration.")
        # Create a dummy image (e.g., a checkerboard or gradient)
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                dummy_img[i, j] = [i, j, (i+j)//2] # Simple gradient
        # Add some "damage" (black squares)
        cv2.rectangle(dummy_img, (50, 50), (100, 100), (0, 0, 0), -1)
        cv2.rectangle(dummy_img, (150, 180), (200, 230), (0, 0, 0), -1)
        damaged_image_np = dummy_img

    damaged_image_np = cv2.cvtColor(damaged_image_np, cv2.COLOR_BGR2RGB) # OpenCV loads BGR
    damaged_image_np = damaged_image_np.astype(np.float32) / 255.0 # Normalize to [0, 1]
    print(f"Loaded image of shape: {damaged_image_np.shape}")
except Exception as e:
    print(f"Error loading image: {e}")
    exit() # Exit if image can't be loaded/created

IMG_HEIGHT, IMG_WIDTH, CHANNELS = damaged_image_np.shape

# --- 2. Data Generator (for on-the-fly patch generation) ---
def get_random_patch(image, patch_size, mask_size):
    """
    Extracts a random patch from the image and applies a synthetic mask.
    Returns the masked patch (input) and the original patch (target).
    """
    # Ensure there's enough room for patch and mask
    if image.shape[0] < patch_size or image.shape[1] < patch_size:
        raise ValueError(f"Image too small for patch size {patch_size}")

    # Randomly select top-left corner of the patch
    y_start = np.random.randint(0, image.shape[0] - patch_size + 1)
    x_start = np.random.randint(0, image.shape[1] - patch_size + 1)

    original_patch = image[y_start : y_start + patch_size,
                           x_start : x_start + patch_size].copy()

    masked_patch = original_patch.copy()

    num_removals = np.random.randint(2, 10)
    for i in range(num_removals):
        # Apply a random square mask within the patch
        random_mask_size = np.random.randint(3, patch_size/2)
        mask_y_start = np.random.randint(0, patch_size - random_mask_size + 1)
        mask_x_start = np.random.randint(0, patch_size - random_mask_size + 1)
        random_noise_region = np.random.rand(random_mask_size, random_mask_size, CHANNELS).astype(np.float32)
        # damaged_image_np[y_start_mask : y_stop_mask+1, x_start_mask : x_stop_mask+1, :] = \
        #         random_noise_region
            
        masked_patch[mask_y_start : mask_y_start + random_mask_size,
                 mask_x_start : mask_x_start + random_mask_size, :] = random_noise_region # Black out

    return masked_patch, original_patch

# Create a TensorFlow Dataset from a generator
def data_generator(image, patch_size, mask_size, batch_size):
    while True:
        inputs = []
        targets = []
        for _ in range(batch_size):
            inp, tar = get_random_patch(image, patch_size, mask_size)
            inputs.append(inp)
            targets.append(tar)
        yield np.array(inputs), np.array(targets)

train_generator = data_generator(damaged_image_np, PATCH_SIZE, MASK_SIZE, BATCH_SIZE)

# --- 3. Simple Autoencoder Model ---
def build_autoencoder(input_shape):
    input_img = keras.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x) # Output: 32x32x32
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x) # Output: 16x16x64

    # Decoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x) # Output: 32x32x64
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x) # Output: 64x64x32
    decoded = layers.Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x) # Output: 64x64x3 (for RGB)

    autoencoder = keras.Model(input_img, decoded)
    return autoencoder

# --- Simple Neural Network Model (2-3 layers max) ---
def build_simple_nn(input_shape):
    input_img = keras.Input(shape=input_shape)

    # Layer 1: Learn initial features
    x = layers.Conv2D(64, (5, 5), activation='relu', padding='same')(input_img) 
    # Layer 2: Refine features
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    # Layer 3 (Output): Predict pixel values
    # The output layer has 'CHANNELS' filters to match the input image's color channels (e.g., 3 for RGB)
    # 'sigmoid' activation ensures output pixel values are between 0 and 1.
    output_img = layers.Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)

    simple_nn_model = keras.Model(input_img, output_img)
    return simple_nn_model

# --- 5. Inpainting (Inference) ---
def inpaint_image(model, image, patch_size, mask_size_for_inference=None):
    inpainted_image = image.copy()
    img_height, img_width, _ = inpainted_image.shape
    
    # Calculate step size. A smaller step size means more overlap and smoother results,
    # but also more computation. For full coverage, make sure it's <= patch_size.
    # For now, let's just do full patches for simplicity.
    step_size = patch_size // 2 if patch_size > 1 else 1 # Example: half overlap

    for ((x_start_mask, y_start_mask), (x_stop_mask, y_stop_mask)) in corruptions:
        corruption_height = y_stop_mask - y_start_mask
        corruption_width = x_stop_mask - x_start_mask

        # Add random noise in the damaged region (test)
        random_noise_region = np.random.rand(corruption_height + 1, corruption_width + 1, CHANNELS).astype(np.float32)
        damaged_image_np[y_start_mask : y_stop_mask +1, x_start_mask : x_stop_mask + 1, :] = \
            random_noise_region


    for y in range(0, img_height - patch_size + 1, step_size):
        for x in range(0, img_width - patch_size + 1, step_size):
            # Extract the current patch from the *original damaged image*
            current_damaged_patch = damaged_image_np[y : y + patch_size, x : x + patch_size].copy()
            
            # Identify the damaged areas within this patch based on the global damage_mask
            # Create a multi-channel mask for the current patch
            current_patch_mask = damage_mask[y : y + patch_size, x : x + patch_size].astype(bool)
            multi_channel_mask_for_patch = np.stack([current_patch_mask]*CHANNELS, axis=-1)

            # If there's damage in this patch, prepare it for the model
            if np.any(current_patch_mask):
                # The input to the model should be the patch with its damaged areas (from `damage_mask`) filled with black.
                # Your `damaged_image_np` already has these regions blacked out.
                # So `current_damaged_patch` is already the correct input if you want the model to predict the missing parts.

                # Predict the inpainted version of this patch
                predicted_patch = model.predict(np.expand_dims(current_damaged_patch, axis=0), verbose=0)[0]

                # Only replace the damaged pixels in the inpainted_image with the predicted pixels
                # Use the multi_channel_mask_for_patch to selectively copy
                inpainted_image[y : y + patch_size, x : x + patch_size][multi_channel_mask_for_patch] = \
                    predicted_patch[multi_channel_mask_for_patch]
            # else:
                # If there's no damage in this patch, we don't need to do anything,
                # as the inpainted_image already contains the original (undamaged) pixels.

    return inpainted_image

# --- 5. Inpainting (Inference) ---
def inpaint_image_by_corruption(model, image, patch_size, corruptions_list):
    """
    Applies the trained model to inpaint the image by iterating over specified corruption regions.
    For each corruption, extracts a patch of `patch_size` around it, predicts, and then
    replaces the damaged area with the corresponding part of the prediction.
    """
    inpainted_image = image.copy() # Start with the damaged image

    mask_size = MASK_SIZE # Assuming MASK_SIZE is the size of the corruptions (32x32)

    # Calculate the padding needed for the context patch around the mask
    # E.g., if PATCH_SIZE=64, MASK_SIZE=32, padding is (64-32)/2 = 16
    padding = (patch_size - mask_size) // 2

    for ((x_start_mask, y_start_mask), (x_stop_mask, y_stop_mask)) in corruptions_list:
        corruption_height = y_stop_mask - y_start_mask
        corruption_width = x_stop_mask - x_start_mask
        # Determine the ideal top-left corner of the 64x64 patch that centers the 32x32 mask
        ideal_patch_y = y_start_mask - padding
        ideal_patch_x = x_start_mask - padding

        # Add random noise in the damaged region (test)
        random_noise_region = np.random.rand(corruption_height+1, corruption_width+1, CHANNELS).astype(np.float32)
        damaged_image_np[y_start_mask : y_stop_mask+1, x_start_mask : x_stop_mask+1, :] = \
            random_noise_region
        
        # Create a temporary buffer for the input patch (64x64) to handle boundary conditions
        input_for_model = np.zeros((patch_size, patch_size, CHANNELS), dtype=np.float32)

        # Calculate coordinates for copying from the damaged image into the input_for_model buffer
        # This handles cases where the patch goes beyond image boundaries
        copy_y_start_img = max(0, ideal_patch_y)
        copy_x_start_img = max(0, ideal_patch_x)
        copy_y_end_img = min(IMG_HEIGHT, ideal_patch_y + patch_size)
        copy_x_end_img = min(IMG_WIDTH, ideal_patch_x + patch_size)

        copy_height = copy_y_end_img - copy_y_start_img
        copy_width = copy_x_end_img - copy_x_start_img

        # Calculate where in the `input_for_model` buffer to paste the copied image data
        # This accounts for the patch being clipped at image boundaries (e.g., if ideal_patch_y is negative)
        paste_y_start_buffer = 0 if ideal_patch_y >= 0 else -ideal_patch_y
        paste_x_start_buffer = 0 if ideal_patch_x >= 0 else -ideal_patch_x

        if copy_height > 0 and copy_width > 0:
            # Copy the relevant region from the *already damaged* image into our input buffer
            # The model expects a blacked-out region, and our `damaged_image_np` already has this.
            input_for_model[paste_y_start_buffer : paste_y_start_buffer + copy_height,
                            paste_x_start_buffer : paste_x_start_buffer + copy_width, :] = \
                damaged_image_np[copy_y_start_img : copy_y_end_img,
                                 copy_x_start_img : copy_x_end_img, :]
        else:
            print(f"Warning: Skipping corruption at ({x_start_mask}, {y_start_mask}) due to invalid patch extraction. "
                  "This usually means the corruption is out of image bounds.")
            continue


        # Perform the prediction on the 64x64 patch
        predicted_patch = model.predict(np.expand_dims(input_for_model, axis=0), verbose=0)[0]

        # Determine the region in the *predicted_patch* that corresponds to the original mask location.
        # This is where the model has filled in the missing pixels.
        # If the patch was perfectly centered, this would be predicted_patch[padding:padding+mask_size, padding:padding+mask_size].
        # We need to adjust for cases where `ideal_patch_y/x` might have been negative (near image border).
        pred_mask_y_start_in_patch = y_start_mask - ideal_patch_y
        pred_mask_x_start_in_patch = x_start_mask - ideal_patch_x
        
        # Extract the relevant part from the predicted patch (this will be 32x32 if possible)
        predicted_mask_region = predicted_patch[
            pred_mask_y_start_in_patch : pred_mask_y_start_in_patch + mask_size,
            pred_mask_x_start_in_patch : pred_mask_x_start_in_patch + mask_size,
            :
        ]
        
        # Place this predicted 32x32 region back into the original image at the corruption's location.
        # Ensure we don't go out of bounds of the original image when pasting.
        paste_y_end_img = min(IMG_HEIGHT, y_start_mask + predicted_mask_region.shape[0])
        paste_x_end_img = min(IMG_WIDTH, x_start_mask + predicted_mask_region.shape[1])

        # Adjust the size of the region to paste if it was clipped by image bounds
        paste_height = paste_y_end_img - y_start_mask
        paste_width = paste_x_end_img - x_start_mask

        if paste_height > 0 and paste_width > 0:
            inpainted_image[y_start_mask : y_start_mask + paste_height,
                            x_start_mask : x_start_mask + paste_width, :] = \
                predicted_mask_region[0:paste_height, 0:paste_width, :]

    return inpainted_image

# --- 5. Inpainting (Inference) ---
def inpaint_image_by_corruption_v2(model, image, patch_size, corruptions_list):
    """
    Fills the damaged areas/corruptions in the image with random noise.

    Args:
        model: The trained inpainting model (not used for filling with noise, but kept for signature consistency).
        image (np.array): The original damaged image (normalized to [0, 1]).
        patch_size (int): The patch size used for training (not directly used for noise filling, but can be ignored).
        corruptions_list (list): A list of tuples, where each tuple specifies
                                 ((x_start, y_start), (x_stop, y_stop)) of a corrupted region.

    Returns:
        np.array: The image with corrupted areas filled with random noise.
    """
    inpainted_image = image.copy() # Start with the damaged image

    for ((x_start, y_start), (x_stop, y_stop)) in corruptions_list:
        # Calculate the dimensions of the current corruption
        corruption_height = y_stop - y_start
        corruption_width = x_stop - x_start

        # Ensure valid dimensions
        if corruption_height <= 0 or corruption_width <= 0:
            print(f"Warning: Skipping corruption at ({x_start}, {y_start}) to ({x_stop}, {y_stop}) due to zero or negative dimensions.")
            continue

        # Generate random noise for this specific corruption region
        # The noise values are in the range [0, 1], matching the normalized image pixel values.
        random_noise_region = np.random.rand(corruption_height, corruption_width, CHANNELS).astype(np.float32)

        # Determine the actual region to paste, accounting for image boundaries
        paste_y_end_img = min(IMG_HEIGHT, y_start + corruption_height)
        paste_x_end_img = min(IMG_WIDTH, x_start + corruption_width)

        paste_height_actual = paste_y_end_img - y_start
        paste_width_actual = paste_x_end_img - x_start

        if paste_height_actual > 0 and paste_width_actual > 0:
            # Place this random noise region back into the inpainted image
            inpainted_image[y_start : y_start + paste_height_actual,
                            x_start : x_start + paste_width_actual, :] = \
                random_noise_region[0:paste_height_actual, 0:paste_width_actual, :]
        else:
            print(f"Warning: Skipping corruption at ({x_start}, {y_start}) to ({x_stop}, {y_stop}) due to zero or negative effective paste dimensions after clipping.")

    return inpainted_image

# To see individual training patch examples
# masked_ex, original_ex = next(train_generator)
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Example Masked Patch (Input)")
# plt.imshow(masked_ex[0])
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.title("Example Original Patch (Target)")
# plt.imshow(original_ex[0])
# plt.axis('off')
# plt.show()


def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

try:
    filename = 'autoencoder_random_noise_100_50_32_2025-05-21 17:24:35.823522.keras'
    model_path = os.path.join(os.path.dirname(__file__), 'model-builds', filename)
    model = keras.models.load_model(model_path, custom_objects={'SSIMLoss': SSIMLoss})
except:
    model = build_autoencoder((PATCH_SIZE, PATCH_SIZE, CHANNELS))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss=SSIMLoss) # Mean Squared Error is common for image tasks
    model.summary()

    # --- 4. Training ---
    # Calculate steps per epoch based on how many batches you want per epoch
    STEPS_PER_EPOCH = 50 # Each epoch will generate 100 * BATCH_SIZE patches
    print(f"Training for {EPOCHS} epochs, with {STEPS_PER_EPOCH * BATCH_SIZE} patches per epoch.")

    history = model.fit(train_generator,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS,
                        verbose=1)

    filename = f'autoencoder_random_noise_{EPOCHS}_{STEPS_PER_EPOCH}_{BATCH_SIZE}_{time.time()}.keras'.replace(' ', '_').replace(':', '.')
    model_path = os.path.join(os.path.dirname(__file__), 'model-builds', filename)
    model.save(model_path)

# Perform inpainting
inpainted_result = inpaint_image(model, damaged_image_np, PATCH_SIZE, corruptions)

# --- Visualization ---
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title("Original Damaged Image (as input for training patches)")
plt.imshow(damaged_image_np)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Inpainted Result (Patch)")
plt.imshow(inpainted_result)
plt.axis('off')

plt.show()