from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
# Open the image
import os

def preprocess(file_path, new_width=300):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist. Please check the path.")
    image = Image.open(file_path)

    # Convert to RGB (in case it's not already)
    image = image.convert('RGB')

    # Convert to a NumPy array
    rgb_array = np.array(image)

    # Resize the image while keeping the aspect ratio
    aspect_ratio = rgb_array.shape[0] / rgb_array.shape[1]
    new_height = int(new_width * aspect_ratio)

    # Resize using PIL
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert the resized image back to a NumPy array
    return np.array(resized_image)

#path to the image
file_path = os.path.join(os.path.dirname(__file__), 'data', 'Lola.jpg')

rgb_array = preprocess(file_path, new_width=300)

# Display the image
plt.imshow(rgb_array)
plt.axis('off')  # Turn off axis labels
plt.show()

U_orig = rgb_array.copy()


#do random inpainting
def random_inpaint(image, level=0.707):
  w,h,c = image.shape
  mask = np.random.random((w,h)) > level
  for channel in range(c):
    image[:,:, channel] = mask*image[:,:, channel]
  return image, mask

# Apply random inpainting and define the forward operator
U_paint, K_I = random_inpaint(rgb_array, level=0.9)
plt.imshow(U_paint)
plt.axis('off')  # Turn off axis labels
plt.show()

#do tv inpainting
alpha = 1
h, w, c = U_paint.shape

def TV(U, h, w):
    # Reshape the flattened U back to its original shape
    U = U.reshape((h, w, 3))

    # Compute the total variation of the image using vectorized operations
    diff_x = U[1:, :-1, :] - U[:-1, :-1, :]
    diff_y = U[:-1, 1:, :] - U[:-1, :-1, :]
    tv = np.sum(np.sqrt(np.sum(diff_x**2, axis=2)) + np.sqrt(np.sum(diff_y**2, axis=2)))
    return tv

def KU(U, K_I, h, w):
    # Reshape the flattened U and K_I back to their original shapes
    U = U.reshape((h, w, 3))
    K_I = K_I.reshape((h, w))

    # Forward operator
    KU = np.copy(U)
    for i in range(3):
        KU[:, :, i] = K_I * U[:, :, i]
    return KU.flatten()

def Objective(U, U_paint, K_I, h, w, alpha=1):
    # Compute the objective function
    return 0.5 * np.linalg.norm(KU(U, K_I, h, w) - U_paint)**2 + alpha * TV(U, h, w)
def Gradient(U, U_paint, K_I, h, w, alpha=1):
    # Compute the gradient of the objective function
    U = U.reshape((h, w, 3))
    KU_U = KU(U, K_I, h, w).reshape((h, w, 3))

    # Compute the gradient of the TV term
    grad_tv = np.zeros_like(U)
    for i in range(h - 1):
        for j in range(w - 1):
            vec1 = U[i + 1, j, :] - U[i, j, :]
            vec2 = U[i, j + 1, :] - U[i, j, :]
            norm_vec1 = np.linalg.norm(vec1)
            norm_vec2 = np.linalg.norm(vec2)
            if norm_vec1 > 0:
                grad_tv[i + 1, j, :] += vec1 / norm_vec1
                grad_tv[i, j, :] -= vec1 / norm_vec1
            if norm_vec2 > 0:
                grad_tv[i, j + 1, :] += vec2 / norm_vec2
                grad_tv[i, j, :] -= vec2 / norm_vec2

    # Compute the gradient of the objective function
    grad_obj = (KU_U.flatten() - U_paint) + alpha * grad_tv.flatten()
    return grad_obj
#optimize
U_0 = U_paint.copy()
U_0 = U_0.flatten()
U_paint = U_paint.flatten()
K_I = K_I.flatten()
res = minimize(Objective, U_0, args=(U_paint, K_I, h, w, alpha), method='L-BFGS-B', jac=Gradient, options={'disp': True})

# Reshape the result back to the original image shape
U = res.x.reshape((h, w, 3))

# Plot the result and compare with the original image
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(U_orig)
plt.title('Original Image')
plt.axis('off')


plt.subplot(1, 3, 2)
plt.imshow(U_paint.reshape((h, w, 3)))
plt.title('Corrupted Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(U)
plt.title('Inpainted Image')
plt.axis('off')
plt.show()


