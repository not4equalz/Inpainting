import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2 # For image loading and manipulation


# --- Configuration ---
PATCH_SIZE = 64  # Size of the square patches (e.g., 64x64 pixels)
MASK_SIZE = 32   # Size of the square mask to apply within the patch
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

corruptions = [[(50, 50), (81, 81)],
 [(150, 180), (181, 211)],
 [(50, 50), (81, 81)],
 [(200, 300), (231, 331)],
 [(400, 100), (431, 131)],
 [(600, 500), (631, 531)],
 [(100, 800), (131, 831)]]

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
            damage_mask[y_start : y_stop, x_start : x_stop] = 1
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

    # Apply a random square mask within the patch
    mask_y_start = np.random.randint(0, patch_size - mask_size + 1)
    mask_x_start = np.random.randint(0, patch_size - mask_size + 1)
    
    # Fill the mask with black (0) for simplicity. Could also be random noise.
    masked_patch[mask_y_start : mask_y_start + mask_size,
                 mask_x_start : mask_x_start + mask_size, :] = 0.0 # Black out

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
    x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    # Layer 3 (Output): Predict pixel values
    # The output layer has 'CHANNELS' filters to match the input image's color channels (e.g., 3 for RGB)
    # 'sigmoid' activation ensures output pixel values are between 0 and 1.
    output_img = layers.Conv2D(CHANNELS, (3, 3), activation='sigmoid', padding='same')(x)

    simple_nn_model = keras.Model(input_img, output_img)
    return simple_nn_model

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

model = build_autoencoder((PATCH_SIZE, PATCH_SIZE, CHANNELS))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mae') # Mean Squared Error is common for image tasks
model.summary()

# --- 4. Training ---
# Calculate steps per epoch based on how many batches you want per epoch
STEPS_PER_EPOCH = 100 # Each epoch will generate 100 * BATCH_SIZE patches
print(f"Training for {EPOCHS} epochs, with {STEPS_PER_EPOCH * BATCH_SIZE} patches per epoch.")

history = model.fit(train_generator,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    epochs=EPOCHS,
                    verbose=1)

# --- 5. Inpainting (Inference) ---
def inpaint_image(model, image, patch_size, mask_size_for_inference=None):
    """
    Applies the trained model to inpaint the entire image.
    This is a simplified approach, often uses sliding windows or full image processing.
    For true inpainting, you would identify the damaged regions and apply the model there.
    Here, we simulate by applying masks and then predicting.
    """
    inpainted_image = damaged_image_np.copy()
    
    # Iterate through the image in chunks (simplified, usually more complex for full image)
    # For a truly damaged image, you'd feed the damaged regions as input
    # For this simple example, we'll just demonstrate by creating a new masked version
    
    # Let's create a test masked version of the original image
    
    # Example: create a large mask in the center for demonstration
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    # test_mask_y_start = center_y - (mask_size_for_inference or MASK_SIZE) // 2
    # test_mask_x_start = center_x - (mask_size_for_inference or MASK_SIZE) // 2
    
    # test_mask_y_end = test_mask_y_start + (mask_size_for_inference or MASK_SIZE)
    # test_mask_x_end = test_mask_x_start + (mask_size_for_inference or MASK_SIZE)
    
    # Ensure mask is within bounds
    # test_mask_y_start = max(0, test_mask_y_start)
    # test_mask_x_start = max(0, test_mask_x_start)
    # test_mask_y_end = min(image.shape[0], test_mask_y_end)
    # test_mask_x_end = min(image.shape[1], test_mask_x_end)
    
    # test_masked_image[test_mask_y_start:test_mask_y_end, 
    #                    test_mask_x_start:test_mask_x_end, :] = 0.0 # Black out for inference

    # To inpaint, you would typically slide a window or pass the entire damaged image
    # For a simple patch-based model, we apply it on a patch containing damage
    

    # Ensure the inference patch is within image bounds
    # y_inf = max(0, min(y_inf, image.shape[0] - PATCH_SIZE))
    # x_inf = max(0, min(x_inf, image.shape[1] - PATCH_SIZE))

    # inference_patch = test_masked_image[y_inf : y_inf + PATCH_SIZE,
    #                                     x_inf : x_inf + PATCH_SIZE]
    
    # Model expects a batch, so add dimension
    # for ((x_start, y_start), (x_stop, y_stop)) in corruptions:
    #     x_inf = max(0, 2*x_start - x_stop)
    #     y_inf = max(0, 2*x_start - x_stop)

    #     predicted_patch = model.predict(np.expand_dims(inference_patch, axis=0))[0]
    #     inpainted_image[y_start : y_stop,
    #                     x_start : x_stop] = predicted_patch
        # Iterate through the image with a sliding window
    # We iterate such that the patch always fits within the image boundaries
    img_height, img_width, _ = inpainted_image.shape
    step_size = max(1, patch_size)

    for y in range(0, img_height - patch_size + 1, step_size):
        for x in range(0, img_width - patch_size + 1, step_size):
            
            # Extract the corresponding mask region for this patch
            current_mask_region = damage_mask[y : y + patch_size, x : x + patch_size].astype(bool)
            if np.any(current_mask_region):
                # Extract the current patch region from the image with damage
                current_patch_region_from_damaged_img = damaged_image_np[y : y + patch_size, x : x + patch_size].copy()
                multi_channel_mask_for_patch = np.stack([current_mask_region]*CHANNELS, axis=-1)
                current_patch_region_from_damaged_img[multi_channel_mask_for_patch] = 0.0

                inpainted_patch = model.predict(np.expand_dims(current_patch_region_from_damaged_img, axis=0), verbose=0)[0]
                inpainted_image[y : y + patch_size, x : x + patch_size][multi_channel_mask_for_patch] = \
                    inpainted_patch[multi_channel_mask_for_patch]
                # inpainted_image[y : y + patch_size, x : x + patch_size][current_mask_region] = inpainted_patch

    # Overlay the predicted patch back onto the image

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
        # Determine the ideal top-left corner of the 64x64 patch that centers the 32x32 mask
        ideal_patch_y = y_start_mask - padding
        ideal_patch_x = x_start_mask - padding

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

# Perform inpainting
inpainted_result = inpaint_image_by_corruption(model, damaged_image_np, PATCH_SIZE, corruptions)

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

# To see individual training patch examples
masked_ex, original_ex = next(train_generator)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Example Masked Patch (Input)")
plt.imshow(masked_ex[0])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Example Original Patch (Target)")
plt.imshow(original_ex[0])
plt.axis('off')
plt.show()