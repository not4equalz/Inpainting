import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
# PARAMETER TUNING
alpha = 0.2
# Width of the image after resizing  set to 1 for no resizing
width = 1

# if random inpainting is used, set corruption level
corruption = 0.8  # Corruption level between 0 (no corruption) and 1 (black pixels only)


#if square inpainting is used, set square_size and num_squares
square_size = 20  # Size of the squares to be inpainted
num_squares = 10 # Number of squares to be inpainted

#if line inpainting is used, set line_width and angle
line_width = 5  # Width of the line to be inpainted
angle = 45  # Angle of the line in degrees

#maximum number of iterations for the optimizer
max_iter = 1000

#path to the image
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Lola.jpg')


# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU
#device = torch.device("cpu")
print(f"Using device: {device}")


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

rgb_array = preprocess(file_path, new_width=width)
U_orig = rgb_array.copy()

# Convert to torch tensor (C, H, W) on the specified device
image = torch.tensor(rgb_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
_, c, h, w = image.shape

# === Random Inpainting Mask ===
def random_inpaint(image, level=0.9):
    mask = (torch.rand((1, 1, h, w), device=image.device) > level).float()
    corrupted = image * mask
    return corrupted, mask

def square_inpaint(image, square_size=50, num_squares=7):
    mask = torch.ones((1, 1, h, w), device=image.device)
    for _ in range(num_squares):
        x = torch.randint(0, w - square_size, (1,), device=image.device).item()
        y = torch.randint(0, h - square_size, (1,), device=image.device).item()
        mask[:, :, y:y + square_size, x:x + square_size] = 0
    corrupted = image * mask
    return corrupted, mask

def line_inpaint(image, line_width=5, angle=0):
    mask = torch.ones((1, 1, h, w), device=image.device)
    center_x, center_y = w // 2, h // 2
    angle_rad = torch.deg2rad(torch.tensor(angle, device=image.device))
    
    # Create a grid of coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=image.device), torch.arange(w, device=image.device), indexing='ij')
    x_rot = (x_coords - center_x) * torch.cos(angle_rad) + (y_coords - center_y) * torch.sin(angle_rad)
    
    # Apply the mask based on line width
    mask[:, :, (x_rot.abs() < line_width)] = 0
    corrupted = image * mask
    return corrupted, mask


#uncomment one of the following lines to use either random inpainting, square inpainting or line impainting
#U_paint, mask = line_inpaint(image, line_width=line_width, angle=angle)
#U_paint, mask = random_inpaint(image, level=corruption)
U_paint, mask = square_inpaint(image, square_size=square_size, num_squares=num_squares)

# === Total Variation (explicit, as in your code) ===
def tv_loss(U):
    dx = U[:, :, 1:, :-1] - U[:, :, :-1, :-1]
    dy = U[:, :, :-1, 1:] - U[:, :, :-1, :-1]
    tv = torch.sum(torch.sqrt(dx**2 + dy**2 + 1e-6))
    return tv / (U.numel() * 1.0)  # Normalize by total number of elements


# === Objective Function ===
U = U_paint.clone().detach().contiguous().requires_grad_(True)

optimizer = torch.optim.LBFGS([U], max_iter=max_iter, lr=1.0, line_search_fn="strong_wolfe")

def closure():
    optimizer.zero_grad()
    fidelity = torch.nn.functional.mse_loss(U * mask, U_paint)
    tv = tv_loss(U)
    loss = fidelity + alpha * tv
    loss.backward()
    print(f"Loss: {loss.item():.6f} | Fidelity: {fidelity.item():.6f} | TV: {tv.item():.6f}")
    return loss

# === Optimization ===
print("Starting optimization...")
optimizer.step(closure)

# === Visualization ===
def to_numpy(tensor):
    return tensor.detach().cpu().squeeze().clamp(0, 1).permute(1, 2, 0).numpy()

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(U_orig)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(to_numpy(U_paint))
corruption_percentage = (1 - mask.mean().item()) * 100
plt.title(f"Corrupted ({corruption_percentage:.2f}%)")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(to_numpy(U))
plt.title(f"Inpainted ({device})")
plt.axis("off")

plt.tight_layout()
plt.show()
