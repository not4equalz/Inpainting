from __future__ import division
import sys
sys.path.append("./lib") # Ensures Python can find modules in the 'lib' directory
from ttictoc import tic,toc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
import Utils.Corruptions as image_corruption # Assuming this module is available
import matplotlib.animation as animation # Import animation module

def corrupt(original, method):
    match method:
        case "random": corrupt, mask = image_corruption.random_inpaint(original, level=0.7)
        case "square": corrupt, mask = image_corruption.square_inpaint(original, square_size=30, num_squares=10)
        case "line":   corrupt, mask = image_corruption.line_inpaint(original, line_width=10, angle=45)
    return corrupt, mask

def harmonic(input_img, mask_img, tol, maxiter, dt, animation_interval=50): # Added animation_interval
    C = input_img.shape[2]
    u = input_img.copy().astype(np.float64)
    frames = [] # To store frames for animation

    for c in range(0,C):
        print(f"Inpainting channel {c+1}/{C}...")
        channel_frames = [] # Store frames for the current channel's progress
        u_channel_initial = input_img[:,:,c].squeeze().copy() # Keep a copy of the initial corrupted channel

        for iter_count in range(0,maxiter):
            u_channel = u[:,:,c].squeeze()
            input_channel = input_img[:,:,c].squeeze() # Original corrupted channel
            mask_channel = mask_img.squeeze()

            laplacian = cv.Laplacian(u_channel, cv.CV_64F)

            # The update rule for inpainting
            unew_channel = u_channel + dt * laplacian * (1-mask_channel)

            diff_u = np.linalg.norm(unew_channel.reshape(-1)-u_channel.reshape(-1),2)/np.linalg.norm(unew_channel.reshape(-1),2);
            u[:,:,c] = unew_channel

            if iter_count % animation_interval == 0 or iter_count == maxiter -1 :
                # Create a full image frame for animation
                # We need to be careful here if we want to show per-channel progress or combined progress
                # For simplicity, let's store the state of 'u' at this point
                # If you want to see the specific channel being inpainted, this logic would need adjustment
                current_frame_image = u.copy()
                # Clamp values to [0, 1] for display
                current_frame_image = np.clip(current_frame_image, 0, 1)
                frames.append(current_frame_image)


            if diff_u < tol:
                print(f"Converged for channel {c+1} after {iter_count+1} iterations.")
                # Add final state if not already added by animation_interval
                if iter_count % animation_interval != 0 and iter_count != maxiter -1:
                    current_frame_image = u.copy()
                    current_frame_image = np.clip(current_frame_image, 0, 1)
                    frames.append(current_frame_image)
                break
        else:
            print(f"Max iterations reached for channel {c+1} ({maxiter} iterations).")
            # Add final state if not already added by animation_interval
            if (maxiter -1) % animation_interval != 0:
                current_frame_image = u.copy()
                current_frame_image = np.clip(current_frame_image, 0, 1)
                frames.append(current_frame_image)


    return u, frames

# --- Main script ---
clean_np_image = mpimg.imread('./data/Lola.jpg') / 255.0


clean_torch_image = torch.from_numpy(clean_np_image).permute(2, 0, 1).unsqueeze(0).float()

# Corruption
corrupted_torch_image, torch_mask = corrupt(clean_torch_image, "line")

corrupted_np_image = corrupted_torch_image.squeeze(0).permute(1, 2, 0).numpy()
mask_np = torch_mask.squeeze(0).permute(1, 2, 0).numpy()

# Inpainting
tic()
# Reduce maxiter for faster animation generation, adjust as needed
inpainted_image, animation_frames = harmonic(
    corrupted_np_image.copy(), # Pass a copy to avoid modifying the original corrupted image directly
    mask_np.copy(),
    fidelity=10,
    tol=1e-5, # Slightly higher tolerance for faster convergence in animation
    maxiter=10000, # Reduced for animation speed
    dt=0.1,
    animation_interval=60
)
elapsed_time = toc()

print(f'Elapsed time for inpainting: {elapsed_time:.4f} seconds')
print(f'Number of frames captured: {len(animation_frames)}')

# --- Create Animation ---
if animation_frames:
    fig_anim = plt.figure(figsize=(8, 8))
    plt.axis('off')
    im_anim = plt.imshow(animation_frames[0]) # Initialize with the first frame

    def update_figure(frame_index):
        im_anim.set_array(animation_frames[frame_index])
        return [im_anim]

    # Create the animation
    # interval is the delay between frames in milliseconds
    anim = animation.FuncAnimation(fig_anim, update_figure, frames=len(animation_frames), interval=30, blit=True)

    # To save the animation, you might need ffmpeg or another writer installed:
    try:
        anim.save('inpainting_animation.gif', writer='pillow', fps=10)
        print("Animation saved as inpainting_animation.gif")
    except Exception as e:
        print(f"Could not save animation (ensure 'pillow' is installed or try 'ffmpeg'): {e}")
        print("Attempting to show animation instead.")
    plt.show() # Show animation if saving fails or not preferred
