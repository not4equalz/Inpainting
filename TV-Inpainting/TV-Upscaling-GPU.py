
import matplotlib.pyplot as plt
import os
import Utils.Preprocessing as pre
import torch
import Utils.loss as lossfunc
from Utils.SSIM import calculate_ssim_arrays
from matplotlib.image import imsave
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#path to the image  
file_path = os.path.join(os.path.dirname(__file__), 'data', 'BBall.png')

#parameters
new_scale = 3 #scale factor for upscaling
iter = 1500 #number of iterations for the optimizer

#preprocess the image
Orig, rgb_scaled, mask = pre.preprocess_upscale(file_path, new_scale=new_scale)
#preprocess the image
Orig, rgb_scaled, mask = pre.preprocess_upscale(file_path, new_scale=new_scale)

# Convert to torch tensor (C, H, W) on the specified device
rgb_scaled_tensor = torch.tensor(rgb_scaled, dtype=torch.float32, device=device).permute(2, 0, 1)  # (C, H, W)
mask_tensor = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)  # (1, H, W)
image = rgb_scaled_tensor.unsqueeze(0)  # (1, C, H, W)
_, c, h, w = image.shape

# === Objective Function ===
U = rgb_scaled_tensor.clone().detach().unsqueeze(0).contiguous().requires_grad_(True)  # Add batch dimension
optimizer = torch.optim.LBFGS([U], max_iter=iter, lr=1.0, line_search_fn="strong_wolfe")
def closure():
    optimizer.zero_grad()

    # Enforce hard constraints in-place
    with torch.no_grad():
        U.data = U.data * (1 - mask_tensor) + rgb_scaled_tensor.unsqueeze(0) * mask_tensor

    fidelity = torch.nn.functional.mse_loss(U * mask_tensor, rgb_scaled_tensor.unsqueeze(0))
    tv = lossfunc.tv_loss(U)
    loss = tv
    loss.backward()
    U.grad *= (1 - mask_tensor)
    print(f"Loss: {loss.item():.6f} | Fidelity: {fidelity.item():.6f} | TV: {tv.item():.6f}")
    return loss
# === Optimization ===
print("Starting optimization...")
optimizer.step(closure)


#convert to numpy for visualization
U = lossfunc.to_numpy(U.squeeze(0))  # Remove batch dimension for visualization
U_paint = lossfunc.to_numpy(rgb_scaled_tensor)


plt.subplot(1, 3, 1)
plt.imshow(Orig)
plt.title("Original Image")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(U_paint)
plt.title(f"Upscaled Image {new_scale}x")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(U)
plt.title("Inpainted Image")
plt.axis('off')
# Save the figure
output_dir = os.path.join(os.path.dirname(__file__), 'generated', 'upscaling')
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'upscaled_result_{new_scale}x.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"Saved figure to: {output_path}")

# Save the inpainted output U as a separate image
U_save_path = os.path.join(output_dir, f'inpainted_output_{new_scale}x.png')
imsave(U_save_path, U, dpi=300)
print(f"Saved inpainted output to: {U_save_path}")

plt.show()
