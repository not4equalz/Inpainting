import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from Utils.SSIM import calculate_ssim_arrays
from matplotlib.animation import FuncAnimation
import Utils.Corruptions as painting
import Utils.loss as lossfunc
import Utils.Preprocessing as pre
np.random.seed(0)
torch.manual_seed(0)
# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Force CPU
#device = torch.device("cpu")
print(f"Using device: {device}")


#path to the image
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Lola.jpg')

# Width of the image after resizing  set to 1 for no resizing
width = 1

inpainting_method = "square"  # Options: "line", "random", "square"

# if random inpainting is used, set corruption level
corruption = 0.98  # Corruption level between 0 (no corruption) and 1 (black pixels only)

#if square inpainting is used, set square_size and num_squares
square_size = 30  # Size of the squares to be inpainted
num_squares = 20 # Number of squares to be inpainted

#if line inpainting is used, set line_width and angle
line_width = 20  # Width of the line to be inpainted
angle = 90  # Angle of the line in degrees

#Number of iterations for the optimizer
iter = 300


#preprocess the image
rgb_array = pre.preprocess(file_path, new_width=width)

#copy the original image for later comparison
U_orig = rgb_array.copy()

# Convert to torch tensor (C, H, W) on the specified device
image = torch.tensor(rgb_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
_, c, h, w = image.shape

# Use the selected inpainting method
if inpainting_method == "line":
    U_paint, mask = painting.line_inpaint(image, line_width=line_width, angle=angle)
elif inpainting_method == "random":
    U_paint, mask = painting.random_inpaint(image, level=corruption)
elif inpainting_method == "square":
    U_paint, mask = painting.square_inpaint(image, square_size=square_size, num_squares=num_squares)
else:
    raise ValueError(f"Invalid inpainting method: {inpainting_method}")

# === Objective Function ===
U = U_paint.clone().detach().contiguous().requires_grad_(True)
optimizer = torch.optim.LBFGS([U], max_iter=iter, lr=1.0, line_search_fn="strong_wolfe")

# Prepare for animation
frames = []

def closure():
    optimizer.zero_grad()

    # Enforce hard constraints in-place
    with torch.no_grad():
        U.data = U.data * (1 - mask) + U_paint * mask

    fidelity = torch.nn.functional.mse_loss(U * mask, U_paint)
    tv = lossfunc.tv_loss(U)
    loss = tv
    loss.backward()
    U.grad *= (1 - mask)
    print(f"Loss: {loss.item():.6f} | Fidelity: {fidelity.item():.6f} | TV: {tv.item():.6f}")

    # Capture the current state of U for animation
    with torch.no_grad():
        frames.append(lossfunc.to_numpy(U).copy())

    return loss

# === Optimization ===
print("Starting optimization...")
optimizer.step(closure)

# Create animation

fig, ax = plt.subplots(dpi=150)  # Increase DPI for higher resolution
im = ax.imshow(frames[0], animated=True)
ax.axis("off")

def update(frame):
    im.set_array(frame)
    return [im]

ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

# Save the animation
corruption_details = f"{inpainting_method}_corruption-{corruption}" if inpainting_method == "random" else \
                     f"{inpainting_method}_square-{square_size}_num-{num_squares}" if inpainting_method == "square" else \
                     f"{inpainting_method}_line-{line_width}_angle-{angle}"
output_path = os.path.join(os.path.dirname(__file__), 'generated', 'animated', f'animation_{corruption_details}.mp4')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
ani.save(output_path, fps=60, extra_args=['-vcodec', 'libx264'])
print(f"Animation saved to: {output_path}")

plt.show()

