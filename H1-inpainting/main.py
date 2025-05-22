import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import Utils.Corruptions as painting
import Utils.loss as lossfunc
import Utils.Preprocessing as pre

np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

file_path = os.path.join(os.path.dirname(__file__), '..', 'TV-Inpainting', 'data', 'Lola.jpg')
width = 1
inpainting_method = "line"
corruption = 0.2
square_size = 20
num_squares = 20
line_width = 20
angle = 90
iter = 600

rgb_array = pre.preprocess(file_path, new_width=width)
U_orig = rgb_array.copy()
image = torch.tensor(rgb_array, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
_, c, h, w = image.shape

if inpainting_method == "line":
    U_paint, mask = painting.line_inpaint(image, line_width=line_width, angle=angle)
elif inpainting_method == "random":
    U_paint, mask = painting.random_inpaint(image, level=corruption)
elif inpainting_method == "square":
    U_paint, mask = painting.square_inpaint(image, square_size=square_size, num_squares=num_squares)
else:
    raise ValueError(f"Invalid inpainting method: {inpainting_method}")


U_initial = U_paint.clone()
U_initial = U_initial + (1 - mask) * torch.rand_like(U_initial) * 0.1 # Small noise in holes

U = U_initial.detach().contiguous().requires_grad_(True) # Now U is initialized with something in the holes

optimizer = torch.optim.LBFGS([U], max_iter=iter, lr=1.0, line_search_fn="strong_wolfe")

# --- Added for visualization of U's evolution ---
frames_dir = os.path.join(os.path.dirname(__file__), 'generated', 'inpainting_frames')
os.makedirs(frames_dir, exist_ok=True)
frame_count = 0
# -------------------------------------------------

def closure():
    global frame_count
    optimizer.zero_grad()

    # --- Crucial Change Here ---
    # We now enforce the hard constraints by keeping the known (mask=1) parts of U fixed
    # to the original corrupted image U_paint. The optimization happens ONLY where mask is 0.
    with torch.no_grad():
        U.data = U.data * (1 - mask) + U_paint * mask
    # ---------------------------

    # Fidelity term: This part usually measures how well the inpainted image matches the known
    # parts of the corrupted image. It acts as a data fidelity term.
    # It seems your original code had fidelity, but wasn't adding it to the 'loss'.
    # For inpainting, you typically have: Loss = Fidelity + Regularization
    fidelity = torch.nn.functional.mse_loss(U * mask, U_paint * mask) # Ensure fidelity only considers known regions

    # L1 norm of the gradient (anisotropic TV)
    grad_x = torch.abs(U[:, :, :, 1:] - U[:, :, :, :-1])
    grad_y = torch.abs(U[:, :, 1:, :] - U[:, :, :-1, :])
    regularization_loss = torch.sum(grad_x) + torch.sum(grad_y)

    # Combine fidelity and regularization. You'll need a lambda parameter to balance them.
    # A larger lambda for regularization encourages smoother results, less for fidelity.
    lambda_reg = 0.01 # Adjust this value! (Hyperparameter tuning)
    loss = fidelity + lambda_reg * regularization_loss

    loss.backward()
    U.grad *= (1 - mask) # Ensure gradients are zero outside the inpainting region
    print(f"Loss: {loss.item():.6f} | Fidelity: {fidelity.item():.6f} | L1 Gradient Loss: {regularization_loss.item():.6f}")

    if frame_count % (iter // 10 + 1) == 0 or frame_count == 0 or frame_count == iter - 1: # Save 10 frames + start/end
        U_np = lossfunc.to_numpy(U.detach().clone())
        plt.imshow(U_np)
        plt.title(f"Iteration {frame_count}")
        plt.axis("off")
        plt.savefig(os.path.join(frames_dir, f"frame_{frame_count:04d}.png"), bbox_inches='tight', dpi=100)
        plt.close()
    frame_count += 1

    return loss

print("Starting optimization...")
optimizer.step(closure)

U = lossfunc.to_numpy(U)
U_paint = lossfunc.to_numpy(U_paint)

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
plt.title(f"Inpainted ({device}){ssim_title}")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(np.clip(U - U_orig, 0, 1))
plt.title(f"Difference")
plt.axis("off")

plt.tight_layout()

if inpainting_method == "line":
    corruption_details = f"line_width_{line_width}_angle_{angle}"
elif inpainting_method == "random":
    corruption_details = f"corruption_{corruption:.2f}"
elif inpainting_method == "square":
    corruption_details = f"square_size_{square_size}_num_squares_{num_squares}"
else:
    corruption_details = "unknown"

output_filename = f"inpainted_image_L1Grad_{inpainting_method}_{corruption_details}.png"
output_path = os.path.join(os.path.dirname(__file__), 'generated', output_filename)

plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Inpainted image saved to: {output_path}")
print(f"Intermediate frames saved to: {frames_dir}")
plt.show()
