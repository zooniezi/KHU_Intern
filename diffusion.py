import torch
import functools

import option as op

def diffusion_coeff(t, sigma):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma ** t, device=op.device)


##### 첫번째 인자 함수의 parameter 중 sigma 를 sigma(25)로 고정시켜놓은 함수를 정의 하는 느낌. 굳이 해야하는지는 모르겠음.

diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=op.sigma)