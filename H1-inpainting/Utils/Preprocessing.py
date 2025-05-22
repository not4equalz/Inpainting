import os
import numpy as np
from PIL import Image
# === Image Preprocessing ===
def preprocess(file_path, new_width=300):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    image = Image.open(file_path).convert('RGB')
    if new_width == 1:
        return np.array(image) / 255.0  # Normalize to [0, 1]
    aspect_ratio = image.height / image.width
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(image) / 255.0  # Normalize to [0, 1]