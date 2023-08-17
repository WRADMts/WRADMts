from WTAN.smoothness_prior import smoothness_norm
import torch
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

def alignment_loss(X_trasformed, thetas, n_channels, WTANargs, gamma):
    '''
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:
        thetas:
        WTANargs:

    Returns:

    '''
    loss = 0
    sim_vec_sum = 0
    prior_loss = 0
    # similarity loss computation

    diff = X_trasformed.unsqueeze(1) - X_trasformed
    row, col, a , b = diff.shape
    diff = diff.reshape(row*col, a, b)
    diff = diff+1e-9
    norm_mat = torch.square(torch.norm(diff, dim=(1,2), p=2))
    sim_vec_sum = -1*(torch.sum(torch.exp(-(norm_mat)/gamma)))
    loss += sim_vec_sum

    # Note: for multi-channel data, assumes same transformation (i.e., theta) for all channels
    if WTANargs.smoothness_prior:
        for theta in thetas:
            # alignment loss penalty for large deformations
            # larger penalty when k increases -> coarse to fine
            prior_loss += 0.1*smoothness_norm(WTANargs.T, theta, WTANargs.lambda_smooth, WTANargs.lambda_var, print_info=False)
        loss += prior_loss
    return loss
