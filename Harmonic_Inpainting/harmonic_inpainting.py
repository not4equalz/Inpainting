from __future__ import division
import sys
sys.path.append("./lib") # Ensures Python can find modules in the 'lib' directory
from ttictoc import tic,toc 
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt  
import numpy as np               
import torch                     
import cv2 as cv
import Utils.Corruptions as image_corruption

def corrupt(original, method):
    match method:
        case "random": corrupt, mask = image_corruption.random_inpaint(original, level=0.9)
        case "square": corrupt, mask = image_corruption.square_inpaint(original, square_size=30, num_squares=10)
        case "line":   corrupt, mask = image_corruption.line_inpaint(original, line_width=10, angle=45)

    return corrupt, mask

def harmonic(input_img,mask_img,tol,maxiter,dt): 
    C = input_img.shape[2]
    u = input_img.copy().astype(np.float64)

    for c in range(0,C):
        print(f"Inpainting channel {c+1}/{C}...")
        for iter in range(0,maxiter):
            u_channel = u[:,:,c].squeeze()
            input_channel = input_img[:,:,c].squeeze()
            mask_channel = mask_img.squeeze()

            laplacian = cv.Laplacian(u_channel, cv.CV_64F)

            unew_channel = u_channel + dt * laplacian * (1-mask_channel)

            # exit condition: Check if the relative change in 'u' is below the tolerance
            diff_u = np.linalg.norm(unew_channel.reshape(-1)-u_channel.reshape(-1),2)/np.linalg.norm(unew_channel.reshape(-1),2);

            u[:,:,c] = unew_channel

            # exit condition
            if diff_u < tol:
                print(f"Converged for channel {c+1} after {iter+1} iterations.")
                break
        else: 
            print(f"Max iterations reached for channel {c+1} ({maxiter} iterations).")

    return u

clean_np_image = mpimg.imread('./data/HFG.jpg') / 255.0
clean_torch_image = torch.from_numpy(clean_np_image).permute(2, 0, 1).unsqueeze(0).float()

# Corruption
corrupted_torch_image, torch_mask = corrupt(clean_torch_image, "line")

corrupted_np_image = corrupted_torch_image.squeeze(0).permute(1, 2, 0).numpy()
mask_np = torch_mask.squeeze(0).permute(1, 2, 0).numpy()

# Inpainting
tic() 
inpainted_image = harmonic(corrupted_np_image, mask_np, tol=1e-5, maxiter=10000, dt=0.1)
elapsed_time = toc()  

print(f'Elapsed time for inpainting: {elapsed_time:.4f} seconds')

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(clean_np_image)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2) 
plt.imshow(corrupted_np_image)
plt.title('Corrupted Image')
plt.axis('off')
plt.subplot(1, 3, 3) 
plt.imshow(inpainted_image)
plt.title('Inpainted Image (Harmonic)')
plt.axis('off')
plt.tight_layout() 
plt.show() 