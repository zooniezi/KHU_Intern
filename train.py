import torch.nn as nn
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import tqdm
import functools

import option as op

# Defining a time-dependent score-based model
class GaussianFourierProjection(nn.Module):  ##### time t를 embading 해줌
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weigh ts during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):  ##### Dense는 input dim -> output dim 으로 dim 맞춰주는 역할
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]
        ##### 뒤에 None 2개는 나중에 broadcasting 될 부분인데, 텐서끼리 더할때(embading된 time t 벡터가 conv한 텐서와 더해질 때)
        ##### 사진의 width, height 차원과 더해질 부분이 없어서 임의로 있다고 생각하기.


class ScoreNet(nn.Module):  ##### U-Net 을 사용함
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        """Initialize a time-dependent score-based network.
        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
                                   nn.Linear(embed_dim, embed_dim))

        ##### encoding 과정에서는 매 스텝마다 conv하면 feature map의 크기는 줄고 channel 수는 2배씩 늘어난다

        # Encoding layers where the resolution decreases
        ##### conv2d: in_channels, out_channels, kernel_size, stride=1, padding=0, dilation =1,groups=True, bias= True, padding_mode='zeros'
        ##### conv2d input : N(batchsize), in_channels, height, width
        ##### conv2d output : N(bathcsize), out_channels, height, width

        ##### groupNorm은 아직 잘 모르겠다.

        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        self.marginal_prob_std = marginal_prob_std  ##### 뭔지 모르겠다 ?????

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))  ##### size [32,256]
        # Encoding path
        h1 = self.conv1(x)  ##### size [32,32,28,28] =[batchsize, channel[0],사진 너비, 사진높이]
        ## Incorporate information from t  ##### 시간정보를 더해줌
        h1 += self.dense1(embed)  ##### size [32,32,28,28]=[32,32,28,28]+[32,32,1,1] by broad casting
        ## Group normalization
        h1 = self.gnorm1(h1)
        ## Activation func
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)  #####여기까지 encoding, conv 하면서 계쏙 spatial 크기는 줄지만, channel의 개수는 점점 늘어남

        # Decoding path
        h = self.tconv4(h4)  ##### tconv는 transpose conv 라고해서 size가 커지는 conv를 저렇게 표현함
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))  ##### h3를 더해주는 형식이 skip connection을 말하는 듯
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        h = h / self.marginal_prob_std(t)[:, None, None, None]
        return h

# Set up the SDE
def marginal_prob_std(t, sigma):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    t = torch.tensor(t, device=op.device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))

# Define the loss function
def loss_fn(model, x, marginal_prob_std, eps=1e-5):
    """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
    random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
    z = torch.randn_like(x)
    std = marginal_prob_std(random_t)
    perturbed_x = x + z * std[:, None, None, None]
    score = model(perturbed_x, random_t)
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=(1, 2, 3)))

    return loss



marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=op.sigma)
# Training
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(op.device)

dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset, batch_size=op.batch_size, shuffle=True, num_workers=0)

optimizer = Adam(score_model.parameters(), lr=op.lr)  ##### parameters()는 신경망 parameter를 optimizer에 전달해 줄 때 사용함.
tqdm_epoch = tqdm.trange(op.n_epochs)

# for epoch in tqdm_epoch: #tqdm_epoch
#   avg_loss = 0.
#   num_items = 0

#   for x, y in data_loader:
#     x = x.to(device)
#     loss = loss_fn(score_model, x, marginal_prob_std_fn)
#     optimizer.zero_grad() ##### gradient 초기화. backward()를 호출할 때마다 변화도가 buffer에 누적되기 때문
#     loss.backward() ##### 역전파 과정에서 각 노드의 변화도를 계산
#     optimizer.step() ##### optimizer(Adam)가 parameter를 업데이트시킴
#     avg_loss += loss.item() * x.shape[0]
#     num_items += x.shape[0]
#
#   # Print the averaged training loss so far.
#   tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
#   # Update the checkpoint after each epoch of training.
#   torch.save(score_model.state_dict(), 'ckpt.pth')

