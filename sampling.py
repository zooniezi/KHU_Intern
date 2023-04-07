import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import circulant
from numpy.linalg import svd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
from torchvision.utils import make_grid
import functools


import option as op
import diffusion
import drift
import train
import samplers


# Sampling
## Load the pre-trained checkpoint from disk.
ckpt = torch.load('checkpt.pth', map_location=op.device)
train.score_model.load_state_dict(ckpt)

## Generate samples using the specified sampler.
sampler = samplers.Euler_Maruyama_sampler  # @param ['Euler_Maruyama_sampler', 'pc_sampler', 'ode_sampler'] {'type': 'raw'}
samples, orig_x, blur_x, degraded_y, f_i_list, x_list, H_i_list, mean_x1 = sampler(train.score_model,
                                                                          train.marginal_prob_std_fn,
                                                                          diffusion.diffusion_coeff_fn,
                                                                          op.sample_batch_size,
                                                                          num_steps = op.num_steps,
                                                                          device=op.device)



## Sample visualization.
samples = samples.clamp(0.0, 1.0)  ##### clamp: 0과 1사이로 rescale 하는 함수인듯

import matplotlib.pyplot as plt

sample_grid = make_grid(samples, nrow=int(np.sqrt(op.sample_batch_size)))

fig = plt.figure(figsize=(9, 9))
rows = 3; cols = 3
ax1 = fig.add_subplot(rows, cols, 2)
ax1.set_title('result')
ax1.axis('off')
ax1.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

original_grid = make_grid(orig_x, nrow=int(np.sqrt(op.batch_size)))
ax2 = fig.add_subplot(rows,cols,1)
ax2.set_title('original image')
ax2.axis('off')
ax2.imshow(original_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

degraded_grid = make_grid(degraded_y, nrow=int(np.sqrt(op.batch_size)))
ax3 = fig.add_subplot(rows,cols,3)
ax3.set_title('degraded image')
ax3.axis('off')
ax3.imshow(degraded_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

f_4_grid = make_grid(f_i_list[400], nrow=int(np.sqrt(op.batch_size)))
ax4 = fig.add_subplot(rows,cols,4)
ax4.set_title('f_4 grid image')
ax4.axis('off')
ax4.imshow(f_4_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

f_5_grid = make_grid(f_i_list[500], nrow=int(np.sqrt(op.batch_size)))
ax5 = fig.add_subplot(rows,cols,5)
ax5.set_title('f_5 grid image')
ax5.axis('off')
ax5.imshow(f_5_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

H_4_grid = make_grid(H_i_list[400], nrow=int(np.sqrt(op.batch_size)))
ax6 = fig.add_subplot(rows,cols,6)
ax6.set_title('H_4 grid image')
ax6.axis('off')
ax6.imshow(H_4_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

H_5_grid = make_grid(H_i_list[500], nrow=int(np.sqrt(op.batch_size)))
ax7 = fig.add_subplot(rows,cols,7)
ax7.set_title('H_5 grid image')
ax7.axis('off')
ax7.imshow(H_5_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

mean_x1_grid = make_grid(mean_x1, nrow=int(np.sqrt(op.batch_size)))
ax8 = fig.add_subplot(rows,cols,8)
ax8.set_title('degraded image')
ax8.axis('off')
ax8.imshow(mean_x1_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)


abs123 = make_grid(abs(samples - mean_x1), nrow=int(np.sqrt(op.batch_size)))
ax9 = fig.add_subplot(rows,cols,9)
ax9.set_title('diff')
ax9.axis('off')
ax9.imshow(abs123.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)

fig.tight_layout()
plt.show()
