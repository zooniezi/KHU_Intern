import torch
import numpy as np

def get_sigmas(device):
    sigma_begin = 190
    sigma_end = 0.01
    num_classes = 1086
    sigmas = torch.tensor(np.exp(np.linspace(np.log(sigma_begin), np.log(sigma_end), num_classes))).float().to(device)

    return sigmas