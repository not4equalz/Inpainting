import os
import numpy as np
from PIL import Image
def preprocess(file_path, new_width=300):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    image = Image.open(file_path).convert('RGB')
    if new_width == 1:
        return np.array(image) / 255.0  
    aspect_ratio = image.height / image.width
    new_height = int(new_width * aspect_ratio)
    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return np.array(image) / 255.0 

def preprocess_upscale(file_path, new_scale=2):
    #scales up the image and keeps the other pixels empty
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    image = Image.open(file_path).convert('RGB')
    image = np.array(image) / 255.0  
    h, w = image.shape[:2]

    new_width = int(w * new_scale)
    new_height = int(h * new_scale)
    newimage = np.zeros((new_height, new_width, 3))  

    mask = np.zeros((new_height, new_width), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            ni = int(i * new_scale)
            nj = int(j * new_scale)
            if ni < new_height and nj < new_width:
                newimage[ni, nj, :] = image[i, j, :]
                mask[ni, nj] = 1  

    return image, newimage, mask
