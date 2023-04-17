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

import diffusion
import drift
import train

import option as op


def Euler_Maruyama_sampler(score_model,
                           marginal_prob_std,
                           diffusion_coeff,
                           sample_batch_size,
                           num_steps,
                           device,
                           eps=1e-3):
    """Generate samples from score-based models with the Euler-Maruyama solver.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation of
        the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
      sample_batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """
    t = torch.ones(sample_batch_size, device=device)

    # torch.randn() : 평균이 0이고 표준편차가 1인 가우시안 정규분포를 이용해 생성
    # init_x = torch.randn(sample_batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    # x = init_x

    # torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    #####
    count = 0;
    cnt = 0
    f_i_list = {}
    H_i_list = {}
    x_list = {}

    for data_x, data_y in train.data_loader:
        if count == 0:
            original_x = data_x
            noise_schedule = torch.ones(sample_batch_size) * op.noise
            Z = torch.randn_like(original_x, device=device) * noise_schedule[:, None, None, None]
            blurred_x = drift.MatMulwithH(H=drift.H, image=data_x, batch_size=sample_batch_size)
            degraded_y = blurred_x
            count += 1
        else:
            break
    #####
    init_x = degraded_y
    x = init_x

    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            cnt += 1
            print(time_step)
            batch_time_step = torch.ones(sample_batch_size, device=device) * time_step  # (1, batchsize)

            #####
            g = diffusion_coeff(batch_time_step)  # (1,batchsize)
            f_i = drift.MatMulwithH(drift.USUt_list[num_steps - cnt + 1], degraded_y, sample_batch_size) - x

            f_i_list[cnt] = f_i
            H_i_list[cnt] = -f_i
            x_list[cnt] = x
            # print(f_i * step_size)

            h_i = -f_i

            # if cnt <= 10:
            #     mean_x = x + step_size * f_i
            # else:
            #     mean_x = x + step_size * f_i + step_size * (g ** 2)[:, None, None, None] * score_model(x, batch_time_step)
            #     mean_x = x + step_size * f_i

            mean_x = x + step_size * f_i + step_size * (g ** 2)[:, None, None, None] * score_model(x, batch_time_step)
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            # x = mean_x
            #####

        VUt = np.dot(np.transpose(drift.Vt), np.transpose(drift.U))
        print(" ",VUt)
        mean_x1 = drift.MatMulwithH(VUt,mean_x,op.sample_batch_size)


    return mean_x, original_x, blurred_x, degraded_y, f_i_list, x_list, H_i_list, mean_x1







def pc_sampler(score_model,
               marginal_prob_std,
               diffusion_coeff,
               sample_batch_size,
               num_steps,
               device,
               snr=op.signal_to_noise_ratio,
               eps=1e-3):
    """Generate samples from score-based models with Predictor-Corrector method.

    Args:
      score_model: A PyTorch model that represents the time-dependent score-based model.
      marginal_prob_std: A function that gives the standard deviation
        of the perturbation kernel.
      diffusion_coeff: A function that gives the diffusion coefficient
        of the SDE.
      sample_batch_size: The number of samplers to generate by calling this function once.
      num_steps: The number of sampling steps.
        Equivalent to the number of discretized time steps.
      device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
      eps: The smallest time step for numerical stability.

    Returns:
      Samples.
    """

    t = torch.ones(sample_batch_size, device=device)
    init_x = torch.randn(sample_batch_size, 1, 28, 28, device=device) * marginal_prob_std(t)[:, None, None, None]
    time_steps = torch.linspace(1., eps, num_steps)
    step_size = time_steps[0] - time_steps[1]

    #####
    cnt = 0;
    count = 0
    f_i_list = {}
    H_i_list = {}
    x_list = {}
    for data_x, data_y in train.data_loader:
        if count == 0:
            original_x = data_x
            noise_schedule = torch.ones(sample_batch_size) * op.noise
            Z = torch.randn_like(original_x, device=device) * noise_schedule[:, None, None, None]
            blurred_x = drift.MatMulwithH(H=drift.H, image=data_x, batch_size=sample_batch_size)
            degraded_y = blurred_x
            count += 1
        else:
            break
    #####

    x = init_x

    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):

            cnt += 1  #####
            print(time_step)  #####

            batch_time_step = torch.ones(sample_batch_size, device=device) * time_step
            # Corrector step (Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2
            x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)

            # Predictor step (Euler-Maruyama)

            #####
            g = diffusion_coeff(batch_time_step)  # (1,batchsize)
            f_i = drift.MatMulwithH(drift.USUt_list[num_steps - cnt + 1], degraded_y, sample_batch_size) - x

            if cnt % 100 == 0:
                f_i_list[cnt] = f_i
                H_i_list[cnt] = -f_i
                x_list[cnt] = x
                print("###########################################")
                print(f_i * step_size)

            h_i = -f_i


            mean_x = x + 3 * step_size * h_i + step_size * (g ** 2)[:, None, None, None] * score_model(x,
                                                                                                       batch_time_step)
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x)
            #####

            # The last step does not include any noise

    return mean_x, original_x, blurred_x, degraded_y, f_i_list, x_list, H_i_list
