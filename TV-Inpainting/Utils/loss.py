import torch
def to_numpy(tensor):
    return tensor.detach().cpu().squeeze().clamp(0, 1).permute(1, 2, 0).numpy()

def tv_loss(U):
    dx = U[:, :, 1:, :-1] - U[:, :, :-1, :-1]
    dy = U[:, :, :-1, 1:] - U[:, :, :-1, :-1]
    tv = torch.sum(torch.sqrt(dx**2 + dy**2 + 1e-6))
    return tv

#normalized version
def tv_loss_normalize(U):
    dx = U[:, :, 1:, :-1] - U[:, :, :-1, :-1]
    dy = U[:, :, :-1, 1:] - U[:, :, :-1, :-1]
    tv = torch.sum(torch.sqrt(dx**2 + dy**2 + 1e-6))
    return tv / (U.numel() * 1.0)  