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


def GenerateGaussianKernel(knl_size, std):
    """
    input: kernel size, standard deviation of Gaussian distribution
    output: Gaussian kernel with size(=kernel size)

    kernel size must be odd number
    """
    # I don't know if kernel's size is (size,size,3) or (size,size)
    Gaussian_2D_kernel = np.zeros([knl_size, knl_size])
    center = (knl_size + 1) / 2

    for i in range(1, knl_size + 1):
        for j in range(1, knl_size + 1):
            # x and y are distance from center
            x = abs(center - j);
            y = abs(center - i)
            Gaussian_2D_kernel[i - 1, j - 1] = 1 / (2 * np.pi * std) * np.exp(-(x ** 2 + y ** 2) / (2 * std ** 2))

    # Normalizaiton for the sum of kernel to be 1
    Gaussian_2D_kernel = Gaussian_2D_kernel / sum(sum(Gaussian_2D_kernel))

    return Gaussian_2D_kernel

def GenerateMatrixH2(image_height, image_width, kernel, knl_size):
    padding_height = image_height - knl_size
    padding_width = image_width - knl_size

    # pad the given kernel
    pad_kernel = np.pad(kernel, ((padding_height, 0), (0, padding_width)), 'constant', constant_values=0)
    pad_knl_height, pad_knl_width = pad_kernel.shape

    # Save H1,H2,...,Hm (m is the num of columns of input signal. it is the same as pad_knl_height)
    # circ_H_list = pad_knl_height x height x width
    circ_H_list = np.zeros((pad_knl_height, image_height, image_width))

    for i in range(1, len(pad_kernel)):
        # circ_H_list[i-1,:,:]
        A = circulant(pad_kernel[-i, :])
        circ_H_list[i - 1] = np.transpose(
            np.concatenate((A[:, -int((knl_size - 1) * 0.5):], A[:, :image_width - int((knl_size - 1) * 0.5)]), axis=1))

    H = np.zeros((pad_knl_height * image_height, pad_knl_height * image_width))

    for i in range(image_height):
        for j in range(image_width):
            # doubly blocked circulant matrix H
            temp = i - j + int((knl_size - 1) * 0.5)
            if temp >= pad_knl_height:
                temp -= pad_knl_height
            H[image_height * i:image_height * (i + 1), image_width * j:image_width * (j + 1)] = circ_H_list[temp]

    return H

def GenerateMatrixH(height, width, kernel, knl_size):
    padding_height = height - knl_size
    padding_width = width - knl_size

    # pad the given kernel
    pad_kernel = np.pad(kernel, ((padding_height, 0), (0, padding_width)), 'constant', constant_values=0)
    pad_knl_height, pad_knl_width = pad_kernel.shape

    # Save H1,H2,...,Hm (m is the num of columns of input signal. it is the same as pad_knl_height)
    # circ_H_list = pad_knl_height x height x width
    circ_H_list = np.zeros((pad_knl_height, height, width))

    for i in range(1, len(pad_kernel)):
        # circ_H_list[i-1,:,:]
        circ_H_list[i - 1] = circulant(pad_kernel[-i, :])

    H = np.zeros((pad_knl_height * height, pad_knl_height * width))

    for i in range(height):
        for j in range(width):
            # doubly blocked circulant matrix H
            H[height * i:height * (i + 1), width * j:width * (j + 1)] = circ_H_list[i - j]

    return H


def FlatImage(image, batch_size):
    """
    Input
    - image that will be flatten
    - batch_size
    Output
    - flatten image whose shape is (batch_size, H*W, 1, channel)
    """
    image = image.permute(0, 2, 3, 1)
    flat_img = torch.flip(image, [2]).view([batch_size, op.img_height * op.img_width, 1, op.color_channel])


    return flat_img


def MatMulwithH2(H, image, batch_size):
    """
    This function can calculate 3-channel(RGB)
    1) We flat the input image. Shape of flatten image is (batch_size, height*width, 1, channel). It's a tensor.
    2) We use 'np.matmul' to multiply [input circulant matrix H] by [each R,G,B channels of flatten image tensor]

    Input
    - H: large matrix will be multiplied flatten image(Ex. Gaussian blur circulant matrix)
    - image
    - batch_size
    Output
    - y: H * image
    """

    flat_x0 = FlatImage(image, batch_size)  # shape: (batch_size, height*width, 1, channel)
    flat_y = torch.zeros_like(flat_x0)  # shape: (batch_size, height*width, 1, channel)

    for i in range(op.color_channel):
        flat_y[:, :, :, i] = np.matmul(H, flat_x0[:, :, :, i])

    print("long vector",flat_y)
    result_y = flat_y.reshape(batch_size, op.img_height, op.img_width, op.color_channel)
    result_y = torch.flip(result_y, [2])
    result_y = result_y.permute(0, 3, 1, 2)
    # y = y.to(torch.uint8)

    return result_y


def Make_USUtDictionary(U, Sigma_vector, num_steps):
    """
    Input
    - U, Sigma_vector
    - num_steps
    Output
    - USUt: dictionary which save many USUt
    """
    USUt = {}
    Sigma_inv = np.reciprocal(Sigma_vector)
    for i in range(1, num_steps + 1):
        temp = np.power(Sigma_vector, float(i-1) / float(num_steps))  # S^((i+1)/N)
        Sigma_i = temp * Sigma_inv  # S^((i+1)/N - 1)
        S_i = np.diag(Sigma_i)
        USUt[i] = np.dot(np.dot(U, S_i), np.transpose(U))  # U*S*Ut

    return USUt



def MatMulwithH(H, image, batch_size):
    """
    This function can calculate 3-channel(RGB)
    1) We flat the input image. Shape of flatten image is (batch_size, height*width, 1, channel). It's a tensor.
    2) We use 'np.matmul' to multiply [input circulant matrix H] by [each R,G,B channels of flatten image tensor]

    Input
    - H: large matrix will be multiplied flatten image(Ex. Gaussian blur circulant matrix)
    - image
    - batch_size
    Output
    - y: H * image
    """

    flat_x0 = FlatImage2(image, batch_size)  # shape: (batch_size, height*width, 1, channel)
    flat_y = torch.zeros_like(flat_x0)  # shape: (batch_size, height*width, 1, channel)

    for i in range(op.color_channel):
        flat_y[:, :, :, i] = np.matmul(H, flat_x0[:, :, :, i])

    # print("long vector",flat_y)

    result_y = flat_y.reshape(batch_size, op.img_height, op.img_width, op.color_channel)
    # result_y = torch.flip(result_y, [2])
    result_y = result_y.permute(0, 3, 1, 2)
    # y = y.to(torch.uint8)

    return result_y

def FlatImage2(image, batch_size):
    """
    Input
    - image that will be flatten
    - batch_size
    Output
    - flatten image whose shape is (batch_size, H*W, 1, channel)
    """
    image = image.permute(0, 2, 3, 1)
    # flat_img = torch.flip(image, [2]).reshape([batch_size, 2 * 2, 1, op.color_channel])
    flat_img = image.reshape([batch_size, op.img_height * op.img_width, 1, op.color_channel])
    return flat_img



Gaussiankernel_2D = GenerateGaussianKernel(knl_size=op.knl_size, std=op.knl_standard_deviation)
H = GenerateMatrixH2(image_height=op.img_height, image_width=op.img_width, kernel=Gaussiankernel_2D, knl_size=op.knl_size)
U, Sigma, Vt = svd(H)
S = np.diag(Sigma)
Sigma_inv = np.reciprocal(Sigma)
S_inv = np.diag(Sigma_inv)
y = MatMulwithH("lena_color_96.jpg",H,1)
USUt_list = Make_USUtDictionary(U=U, Sigma_vector=Sigma, num_steps=op.num_steps)


