from skimage.metrics import structural_similarity as ssim
import cv2
#This file contains a function that calculates the SSIM (Structural Similarity Index) between two images.

def calculate_ssim_paths(image1, image2):
     #calculate SSIM given two image paths
    # Read the images
    img1 = cv2.imread(image1)
    img2 = cv2.imread(image2)

    min_dim = min(image1.shape[0], image1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  


    #calculate SSIM
    ssim_value, _ = ssim(img1, img2, full=True, win_size=win_size, channel_axis=2, data_range=img2.max() - img2.min())
    return ssim_value
def calculate_ssim_arrays(image1, image2):
    #calculate SSIM given two image arrays
    min_dim = min(image1.shape[0], image1.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)  #win_size must be odd and <= min_dim


    #calculate SSIM
    ssim_value, _ = ssim(image1, image2, full=True, win_size=win_size, channel_axis=2, data_range=image2.max() - image2.min())
    return ssim_value


