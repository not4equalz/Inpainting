import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import cv2 # For image loading and manipulation
import os

# --- Configuration ---
PATCH_SIZE = 64  # Size of the square patches (e.g., 64x64 pixels)
MASK_SIZE = 16   # Size of the square mask to apply within the patch (for training)
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

corruptions = [
    [(50, 50), (100, 100)],
    [(150, 180), (200, 230)],
    [(50, 50), (150, 150)],
    [(200, 300), (300, 400)],
    [(400, 100), (500, 200)],
    [(600, 500), (700, 600)],
    [(100, 800), (200, 900)],
]


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
            damage_mask[x_start : x_stop, y_start : y_stop] = 1
        if damaged_image_np is None:
            raise FileNotFoundError # Handle if image path is invalid
    except FileNotFoundError:
        print(f"'{damaged_image_path}' not found. Creating a dummy damaged image for demonstration.")
        # Create a dummy image (e.g., a checkerboard or gradient)
        dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
        for i in range(256):
            for j in range(256):
                dummy_img[i, j] = [i, j, (i+j)//2] # Simple gradient
        # Add some initial "damage" (black squares) to the base image.
        # This will be *part* of the input to the inpainting function,
        # and the arbitrary mask will define *new* areas to inpaint.
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

model = build_autoencoder((PATCH_SIZE, PATCH_SIZE, CHANNELS))
model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss='mse') # Mean Squared Error is common for image tasks
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
def inpaint_image(model, image_with_damage, arbitrary_mask, patch_size):
    """
    Applies the trained model to inpaint an entire image based on an arbitrary mask.

    Args:
        model: The trained Keras autoencoder model.
        image_with_damage: The image with masked/damaged pixels (normalized to [0, 1]).
                           These damaged areas should correspond to the 'arbitrary_mask'.
        arbitrary_mask: A binary NumPy array of the same spatial shape as 'image_with_damage' (HxW).
                        Pixels with True (or 1) indicate damaged regions to be inpainted.
                        Pixels with False (or 0) indicate intact regions.
        patch_size: The size of the patches the model was trained on.

    Returns:
        The inpainted image.
    """
    inpainted_image = image_with_damage.copy()
    img_height, img_width, _ = image_with_damage.shape

    # Validate mask
    if arbitrary_mask.ndim == 3:
        # If mask is HxWxC, convert to HxW by checking if any channel is True
        arbitrary_mask = arbitrary_mask[:, :, 0] > 0.5 # Take one channel and threshold
    
    if arbitrary_mask.shape != (img_height, img_width):
        raise ValueError(f"Arbitrary mask spatial shape {arbitrary_mask.shape} does not match "
                         f"image spatial shape {(img_height, img_width)}. "
                         f"Mask should be HxW (e.g., boolean or 0/1 array).")
    
    # Ensure mask is boolean for easier indexing
    arbitrary_mask = arbitrary_mask.astype(bool)

    # Determine step size for sliding window. A smaller step size helps with blending
    # and ensures full coverage of complex masks, but is slower.
    # Set to a quarter of patch size, or 1 if patch_size is too small.
    step_size = max(1, patch_size // 4)
    
    print(f"\nInpainting with a sliding window of patch size {patch_size} and step size {step_size}...")

    # Iterate through the image with a sliding window
    # We iterate such that the patch always fits within the image boundaries
    for y in range(0, img_height - patch_size + 1, step_size):
        for x in range(0, img_width - patch_size + 1, step_size):
            
            # Extract the current patch region from the image with damage
            current_patch_region_from_damaged_img = image_with_damage[y : y + patch_size, x : x + patch_size].copy()
            
            # Extract the corresponding mask region for this patch
            current_mask_region = arbitrary_mask[y : y + patch_size, x : x + patch_size]

            # Only process if this patch region contains any masked pixels
            if np.any(current_mask_region):
                # Create the input patch for the model:
                # This input patch needs to have its masked areas blacked out for the model.
                input_patch_for_model = current_patch_region_from_damaged_img.copy()
                
                # Expand the 2D mask patch to 3 channels for broadcasting with RGB image data
                multi_channel_mask_for_patch = np.stack([current_mask_region]*CHANNELS, axis=-1)
                
                # Black out the masked regions within this input patch
                input_patch_for_model[multi_channel_mask_for_patch] = 0.0

                # Predict the inpainted patch
                predicted_patch = model.predict(np.expand_dims(input_patch_for_model, axis=0), verbose=0)[0]

                # Update the inpainted_image:
                # Crucially, we only update the *masked parts* of the image
                # with the *predicted values* from the model.
                # Intact parts of the image within the patch are preserved.
                inpainted_image[y : y + patch_size, x : x + patch_size][multi_channel_mask_for_patch] = \
                    predicted_patch[multi_channel_mask_for_patch]
    
    return inpainted_image

# --- Generate a more complex arbitrary mask for demonstration ---
print("\nGenerating a complex arbitrary mask for demonstration...")
arbitrary_mask_np = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

# Add some arbitrary shapes to the mask (these will be the areas to inpaint)
# cv2.circle(arbitrary_mask_np, (IMG_WIDTH // 4, IMG_HEIGHT // 4), 30, 1, -1) # Filled circle
# cv2.rectangle(arbitrary_mask_np, (IMG_WIDTH // 2 - 20, IMG_HEIGHT // 2 + 10),
#                               (IMG_WIDTH // 2 + 40, IMG_HEIGHT // 2 + 60), 1, -1) # Filled rectangle
# cv2.line(arbitrary_mask_np, (IMG_WIDTH // 8, IMG_HEIGHT - 50), (IMG_WIDTH - 50, IMG_HEIGHT // 8), 1, 5) # Thick line
# cv2.circle(arbitrary_mask_np, (IMG_WIDTH - 60, IMG_HEIGHT // 3), 40, 1, 2) # Circle outline
# cv2.ellipse(arbitrary_mask_np, (IMG_WIDTH // 3, IMG_HEIGHT - 60), (50, 20), 45, 0, 360, 1, -1) # Filled ellipse

# Create the actual input image for the inpainting function.
# This image starts with the base 'damaged_image_np' (which might have initial damage)
# and then has the 'arbitrary_mask_np' areas blacked out to simulate missing data.
image_for_inpainting_input = damaged_image_np.copy()
image_for_inpainting_input[arbitrary_mask_np.astype(bool)] = 0.0 # Black out according to the arbitrary mask

# Perform inpainting using the new function
inpainted_result = inpaint_image(model, image_for_inpainting_input, arbitrary_mask_np, PATCH_SIZE)

# --- Visualization ---
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.title("Original Base Image (e.g., for training)")
plt.imshow(damaged_image_np) # This shows the image with initial dummy damage
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title(f"Damaged Image for Inpainting (with arbitrary mask)")
plt.imshow(image_for_inpainting_input) # This is the image the inpainter received as 'image_with_damage'
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Inpainted Result")
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