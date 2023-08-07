from DTAN.smoothness_prior import smoothness_norm
import torch
from sklearn.metrics.pairwise import rbf_kernel, pairwise_kernels
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt

def alignment_loss(X_trasformed, thetas, n_channels, DTANargs, gamma):
    '''
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:
        thetas:
        DTANargs:

    Returns:

    '''
    loss = 0
    sim_vec_sum = 0
    prior_loss = 0
    # X_weight_mean=np.empty((0,60))
    # X_weight_mean=torch.empty(len(X_trasformed),90).cuda()
    inter_loss=0
    if n_channels == 1:
        # Single channel variance across samples

        X_trasformed_np=X_trasformed.detach().cpu().numpy().squeeze() 
        dist_mat=scipy.spatial.distance.cdist(X_trasformed_np,X_trasformed_np)
        width=np.median(dist_mat)
        covar = rbf_kernel(X_trasformed_np,gamma=1/(2*width*width))
        # covar = rbf_kernel(X_trasformed_np,gamma=1/(2*100))
        covar[covar<0.4]=0
        for i in range(covar.shape[1]):
            X_weight_mean=np.append(X_weight_mean,np.array([np.matmul(covar[i],X_trasformed.detach().cpu().numpy().squeeze(axis=1))])/np.sum(covar[i]),axis=0)
        loss+=torch.linalg.norm((X_trasformed.squeeze(axis=1)-torch.Tensor(X_weight_mean).cuda()), ord=2, dim=1).mean() 

        # for i in range(X_trasformed.shape[0]):
        #     # print("vi-vj2",-torch.diagonal(torch.mm((X_trasformed.squeeze(axis=1)-X_trasformed.squeeze(axis=1)[i]),(X_trasformed.squeeze(axis=1)-X_trasformed.squeeze(axis=1)[i]).t()),0))
        #     X_weight_mean_inter = torch.div(torch.exp(-torch.norm(X_trasformed.squeeze(axis=1)-X_trasformed.squeeze(axis=1)[i], dim=1)),2*0.005)
        #     # X_weight_mean_inter = torch.div(torch.exp(-torch.diagonal(torch.mm((X_trasformed.squeeze(axis=1)-X_trasformed.squeeze(axis=1)[i]),(X_trasformed.squeeze(axis=1)-X_trasformed.squeeze(axis=1)[i]).t()),0)),2*1)
        #     # print("sim of j with vi",X_weight_mean_inter.shape)
        #     # print("weight sum",torch.sum(X_weight_mean_inter))
        #     # print("weighted mean",torch.matmul(X_weight_mean_inter,X_trasformed.squeeze(axis=1)))
        #     # inter_loss += torch.linalg.norm((X_trasformed.squeeze(axis=1)[i]-torch.div(torch.matmul(X_weight_mean_inter,X_trasformed.squeeze(axis=1)),torch.sum(X_weight_mean_inter))), ord=2) 
        #     inter_loss += torch.linalg.norm((X_trasformed.squeeze(axis=1)[i]-torch.matmul(X_weight_mean_inter,X_trasformed.squeeze(axis=1))), ord=2) 
        #     # print("inter_loss",inter_loss)    
        # loss+=inter_loss/(len(X_trasformed))
        # # print("loss",loss)

        # X_mean=X_trasformed.mean(dim=0)
        # loss+=torch.linalg.norm((X_trasformed-X_mean), ord=1, dim=0).mean()
        # loss += X_trasformed.var(dim=0, unbiased=False).mean()
    else:
        # variance between signals in each channel (dim=1)
        # mean variance of all channels and samples (dim=0)

        # X_trasformed_np=X_trasformed.detach().cpu().numpy()

        # for j in range(n_channels):
        #     per_channel_loss = 0 
        #     X_weight_mean=np.empty((0,60))
        #     dist_mat=scipy.spatial.distance.cdist(X_trasformed_np[:,j,:],X_trasformed_np[:,j,:])+1e-8
        #     width=np.median(dist_mat)
        #     covar = rbf_kernel(X_trasformed_np[:,j,:],gamma=1/(2*width*width))
        #     # covar = rbf_kernel(X_trasformed_np,gamma=1/(2*100))
        #     # covar[covar<0.4]=0
        #     for i in range(covar.shape[1]):
        #         X_weight_mean=np.append(X_weight_mean,np.array([np.matmul(covar[i],X_trasformed_np[:,j,:])])/np.sum(covar[i]),axis=0)
        #     per_channel_loss+=torch.linalg.norm((X_trasformed[:,j,:]-torch.Tensor(X_weight_mean).cuda()), ord=2, dim=1).mean()

        # loss+=per_channel_loss/(n_channels)

        diff = X_trasformed.unsqueeze(1) - X_trasformed
        row, col, a , b = diff.shape
        diff = diff.reshape(row*col, a, b)
        diff = diff+1e-9
        norm_mat = torch.square(torch.norm(diff, dim=(1,2), p=2))
        sim_vec_sum = -1*(torch.sum(torch.exp(-(norm_mat)/gamma)))
        # norm_mat = torch.norm(torch.norm(diff,dim=1), dim=1, p=1)
        # sim_vec_sum = -1*(torch.sum(torch.exp(-(norm_mat)/gamma)) - 1 * gamma)
        # sim_vec_sum = torch.sum(1-torch.exp(-norm_mat/gamma))  + 0.1 * gamma
        loss += sim_vec_sum

        # X_mean=X_trasformed.mean(dim=0)
        # per_channel_loss=torch.linalg.norm((X_trasformed-X_mean), ord=1, dim=1).mean(dim=0)

        # per_channel_loss = X_trasformed.var(dim=1, unbiased=False).mean(dim=0)
        # per_channel_loss = per_channel_loss.mean()
        # loss += per_channel_loss

    # Note: for multi-channel data, assues same transformation (i.e., theta) for all channels
    if DTANargs.smoothness_prior:
        for theta in thetas:
            # alignment loss takes over variance loss
            # larger penalty when k increases -> coarse to fine
            prior_loss += 0.1*smoothness_norm(DTANargs.T, theta, DTANargs.lambda_smooth, DTANargs.lambda_var, print_info=False)
        loss += prior_loss
    return loss
