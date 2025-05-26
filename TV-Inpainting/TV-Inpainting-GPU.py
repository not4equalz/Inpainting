import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from Utils.SSIM import calculate_ssim_arrays
import Utils.Corruptions as painting
import Utils.loss as lossfunc
import Utils.Preprocessing as pre
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Force CPU
#device = torch.device("cpu")

print(f"Using device: {device}")


#path to the image
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Lola.jpg')

#Width of the image after resizing  set to 1 for no resizing
width = 1

inpainting_method = "line"  # Options: "line", "random", "square"

# if random inpainting is used, set corruption level
corruption = 0.2  # Corruption level between 0 (no corruption) and 1 (black pixels only)

#if square inpainting is used, set square_size and num_squares
square_size = 20
num_squares = 20

#if line inpainting is used, set line_width and angle
line_width = 20
angle = 90

#Number of iterations for the optimizer
iter = 600


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
def closure():
    optimizer.zero_grad()

    #Enforce hard constraints
    with torch.no_grad():
        U.data = U.data * (1 - mask) + U_paint * mask

    fidelity = torch.nn.functional.mse_loss(U * mask, U_paint)
    tv = lossfunc.tv_loss(U)
    loss =tv
    loss.backward()
    U.grad *= (1 - mask)
    print(f"Loss: {loss.item():.6f} | Fidelity: {fidelity.item():.6f} | TV: {tv.item():.6f}")
    return loss

#optimizing step
print("Starting optimization...")
optimizer.step(closure)


#convert to numpy for visualization
U = lossfunc.to_numpy(U)
U_paint = lossfunc.to_numpy(U_paint)


#Compute SSIM between two images
score = calculate_ssim_arrays(U, U_orig)
print(f"Image similarity (SSIM): {score * 100:.2f}%")


plt.figure(figsize=(16, 4))
plt.subplot(1, 4, 1)
plt.imshow(U_orig)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(U_paint)
corruption_percentage = (1 - mask.mean().item()) * 100
print(f"Corruption percentage: {corruption_percentage:.2f}%")
plt.title(f"Corrupted ({corruption_percentage:.2f}%)")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(U)
plt.title(f"Inpainted ({device}) | SSIM: {score * 100:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(np.clip(U - U_orig, 0, 1))
plt.title(f"Difference")
plt.axis("off")

plt.tight_layout()

# Save the inpainted image
if inpainting_method == "line":
    corruption_details = f"line_width_{line_width}_angle_{angle}"
elif inpainting_method == "random":
    corruption_details = f"corruption_{corruption:.2f}"
elif inpainting_method == "square":
    corruption_details = f"square_size_{square_size}_num_squares_{num_squares}"
else:
    corruption_details = "unknown"

output_filename = f"inpainted_image_{inpainting_method}_{corruption_details}.png"
output_path = os.path.join(os.path.dirname(__file__), 'generated', output_filename)

plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Inpainted image saved to: {output_path}")
plt.show()

