import torch


# === Random Inpainting Mask ===
def random_inpaint(image, level=0.9):
    _, _, h, w = image.shape
    mask = (torch.rand((1, 1, h, w), device=image.device) > level).float()
    corrupted = image * mask
    return corrupted, mask
def square_inpaint(image, square_size=50, num_squares=7):
    _, _, h, w = image.shape
    mask = torch.ones((1, 1, h, w), device=image.device)
    for _ in range(num_squares):
        x = torch.randint(0, w - square_size, (1,), device=image.device).item()
        y = torch.randint(0, h - square_size, (1,), device=image.device).item()
        mask[:, :, y:y + square_size, x:x + square_size] = 0
    corrupted = image * mask
    return corrupted, mask
def line_inpaint(image, line_width=5, angle=0):
    _, _, h, w = image.shape
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