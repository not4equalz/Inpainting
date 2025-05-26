import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import Utils.Preprocessing as pre
import Utils.SSIM as ssim
np.random.seed(0)  

#path to the image
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Lola.jpg')

#preprocess the image
rgb_array = pre.preprocess(file_path, new_width=1)

#copy the original image for later comparison
U_orig = rgb_array.copy()

#create noise
noise = np.random.normal(0, .2, rgb_array.shape)
#add noise to the image
U_noisy = rgb_array.copy() + noise
#clip the values to [0, 1]
U_noisy = np.clip(U_noisy, 0, 1)

#second image
U_noisy2 = rgb_array.copy()
size = 40
# add black bars along all four edges (top, bottom, left, right)
U_noisy2[0:size, :, :] = 0        # Top edge
U_noisy2[-size:, :, :] = 0        # Bottom edge
U_noisy2[:, 0:size, :] = 0        # Left edge
U_noisy2[:, -size:, :] = 0        # Right edge
mse1 = np.mean((U_noisy - U_orig) ** 2)
mse2 = np.mean((U_noisy2 - U_orig) ** 2)
ssim1 = ssim.calculate_ssim_arrays(U_noisy, U_orig)
ssim2 = ssim.calculate_ssim_arrays(U_noisy2, U_orig)

plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
plt.imshow(U_orig)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(U_noisy)
plt.title(f"Noisy Image\nMSE: {mse1:.4f}\nSSIM: {ssim1:.4f}")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(U_noisy2)
plt.title(f"Image with Black Bars\nMSE: {mse2:.4f}\nSSIM: {ssim2:.4f}")
plt.axis('off')

print(f"MSE between original and noisy image: {mse1:.4f}")
print(f"MSE between original and noisy image with black bars: {mse2:.4f}")

# save in generated/MSE folder
save_path = os.path.join(os.path.dirname(__file__), 'generated', 'MSE')
if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.savefig(os.path.join(save_path, 'MSE.png'), bbox_inches='tight')
plt.show()


