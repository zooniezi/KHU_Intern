import sys
import numpy as np
import os

import torch
from torchvision.utils import save_image
from models.ncsnv2 import NCSNv2Deeper
from models.ema import EMAHelper
from models import get_sigmas

def main():
    # set random seed (for reproducibility)
    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Set output save directory
    out_dir = "images_samples/images/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Parameters originally from config file
    use_ema = True
    batch_size = 1
    image_size = (3, 128, 128)
    n_steps_each = 3
    step_lr = 0.0000018
    denoise = True

    # Set Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Initializing & Loading the model
    states = torch.load(
        os.path.join('pretrained', 'lsun', 'bedroom', 'checkpoint_150000.pth'),
        map_location=device)

    score = NCSNv2Deeper(image_size[0], device).to(device)
    score = torch.nn.DataParallel(score)
    score.load_state_dict(states[0], strict=True)

    # Set up the ema
    if use_ema:
        ema_rate = 0.999
        ema_helper = EMAHelper(mu=ema_rate)
        ema_helper.register(score)
        ema_helper.load_state_dict(states[-1])
        ema_helper.ema(score)

    # Set up the sigmas
    sigmas_th = get_sigmas(device)
    sigmas = sigmas_th.cpu().numpy()

    score.eval() # Set the model to evaluation mode

    # Noise init
    channels, height, width = image_size
    init_samples = torch.rand(batch_size, channels, height, width, device=device)

    # Generating the samples with Anneal Langevin Dynamics
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(init_samples.shape[0], device=init_samples.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2 # Only used for generating image sample (use when necessary)
            for s in range(n_steps_each):
                grad = score(init_samples, labels)
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                print("level: {}, grad_norm: {}".format(c, grad_norm))

                # This part is for updating the noise and generating image sample (use when necessary)
                noise = torch.randn_like(init_samples)
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                init_samples = init_samples + step_size * grad + noise * np.sqrt(step_size * 2)
                
                image_norm = torch.norm(init_samples.view(init_samples.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

        # This part is for updating the noise and generating image sample (use when necessary)
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(init_samples.shape[0], device=init_samples.device)
            last_noise = last_noise.long()
            init_samples = init_samples + sigmas[-1] ** 2 * score(init_samples, last_noise)
            all_samples = [init_samples.to('cpu')]


    # Saving samples (Only used when the image samples are generated, use when necessary
    sample = all_samples[-1].view(all_samples[-1].shape[0], channels, height, width)
    sample = torch.clamp(sample, 0.0, 1.0)
    save_image(sample, os.path.join(out_dir, 'image_grid_150000.png'))
    torch.save(sample, os.path.join(out_dir, 'samples_150000.pth'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
