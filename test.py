import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *

import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn.functional as F


from util.data import *
from util.preprocess import *


def plot_mean_signal(X_aligned_within_class, X_within_class, ratio, class_num, N=10, dataset_name = "artificial"):

    #check data dim
    if len(X_aligned_within_class.shape) < 3:
        X_aligned_within_class = np.expand_dims(X_aligned_within_class, axis=-1)

    # data dims: (number of samples, dim, channels)
    n_signals = len(X_within_class)  # number of samples within class

    # Sample random signals
    input_shape = X_within_class.shape[1:]  # (channels, dims) PyTorch
    signal_len = input_shape[1]
    n_channels = input_shape[0]

    np.random.seed(2021)
    # indices = np.random.choice(n_signals, N)  # N samples
    # X_within_class = X_within_class[indices, :, :]  # get N samples, all channels
    # X_aligned_within_class = X_aligned_within_class[indices, :, :]

    X_within_class = X_within_class[:, :, :]  # get N samples, all channels
    X_aligned_within_class = X_aligned_within_class[:, :, :]

    # Compute mean signal and variance
    X_mean_t = np.mean(X_aligned_within_class, axis=0)
    X_std_t = np.std(X_aligned_within_class, axis=0)
    upper_t = X_mean_t + X_std_t
    lower_t = X_mean_t - X_std_t

    X_mean = np.mean(X_within_class, axis=0)
    X_std = np.std(X_within_class, axis=0)
    upper = X_mean + X_std
    lower = X_mean - X_std

    # set figure size
    # [w, h] = ratio  # width, height
    # f = plt.figure(1)
    # plt.style.use('seaborn-darkgrid')
    # f.set_size_inches(w, h)
    # f.set_size_inches(w, n_channels * h)


    title_font = 18
    rows = 2
    cols = 2
    # plot each channel
    for channel in range(n_channels):
        [w, h] = ratio  # width, height
        f = plt.figure(1)
        plt.style.use('seaborn-darkgrid')
        f.set_size_inches(w, h)
        plot_idx = 1
        t = range(input_shape[1])
        # Misaligned Signals
        # if channel == 0:
            # ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1 = f.add_subplot(rows, cols, plot_idx)
        ax1.plot(X_within_class[:, channel,:].T)
        plt.tight_layout()
        plt.xlim(0, signal_len)

        if n_channels == 1:
            #plt.title("%d random test samples" % (N))
            plt.title("Misaligned signals", fontsize=title_font)
        else:
            plt.title("Channel: %d, %d random test samples" % (channel, N))
        plot_idx += 1

        # Misaligned Mean
        # if channel == 0:
        #     ax2 = f.add_subplot(rows, cols, plot_idx)
        ax2 = f.add_subplot(rows, cols, plot_idx)
        if n_channels == 1:
            ax2.plot(t, X_mean[channel], 'r',label=f'Average signal-channel:{channel}')
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")
        else:
            ax2.plot(t, X_mean[channel,:], label=f'Average signal-channel:{channel}')
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")

        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.xlim(0, signal_len)

        if n_channels ==1:
            plt.title("Misaligned average signal", fontsize=title_font)
        else:
            plt.title(f"Channel: {channel}, Test data mean signal ({N} samples)")
            ax2.fill_between(t, upper[channel], lower[channel], color='r', alpha=0.2, label=r"$\pm\sigma$")

        plot_idx += 1


        # Aligned signals
        # if channel == 0:
        #     ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3 = f.add_subplot(rows, cols, plot_idx)
        ax3.plot(X_aligned_within_class[:, channel,:].T)
        plt.title("DTAN aligned signals", fontsize=title_font)
        plt.xlim(0, signal_len)

        plot_idx += 1

        # Aligned Mean
        # if channel == 0:
        #     ax4 = f.add_subplot(rows, cols, plot_idx)
        ax4 = f.add_subplot(rows, cols, plot_idx)
        # plot transformed signal
        ax4.plot(t, X_mean_t[channel,:], label=f'Average signal-channel:{channel}')
        if n_channels == 1:
            ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")
        else:
            ax4.fill_between(t, upper_t[channel], lower_t[channel], color='#539caf', alpha=0.6, label=r"$\pm\sigma$")

        plt.legend(loc='upper right', fontsize=12, frameon=True)
        plt.title("DTAN average signal", fontsize=title_font)
        plt.xlim(0, signal_len)
        plt.tight_layout()
        plot_idx += 1


        # plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font+2)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    plt.suptitle(f"{dataset_name}: class-{class_num}", fontsize=title_font+2)

def test(model, dataloader, mode = "test"):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_reconstructed_list = []
    test_ground_list_recons = []
    test_ground_list_pred = []
    test_labels_list = []
    X_aligned_within_class_1 = []

    t_test_predicted_list = []
    t_test_reconstructed_list = []
    t_test_ground_list_recons = []
    t_test_ground_list_pred = []
    t_test_labels_list = []
    # X_aligned_within_class_list = np.array([])

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels_attack, labels_recons, edge_index in dataloader:
        x, y, labels_attack, labels_recons, edge_index = [item.to(device).float() for item in [x, y, labels_attack, labels_recons, edge_index]]
        
        with torch.no_grad():
            # reconstructed, predicted, att_weight = model(x, edge_index, mode) 
            reconstructed, predicted, att_weight, thetas, X_aligned_within_class, gamma = model(x, edge_index, mode) 

            loss_recons = loss_func(reconstructed.float().to(device), x)
            loss_pred = loss_func(predicted.float().to(device), y)
            loss = 1*loss_pred+1*loss_recons
            # loss = loss_func(predicted.float().to(device), y, att_weight)##Pred       

            labels_attack = labels_attack.unsqueeze(1).repeat(1, predicted.shape[1])
            # labels_recons = labels_recons.unsqueeze(1).repeat(1, reconstructed.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_reconstructed_list = reconstructed
                t_test_ground_list_recons = x
                t_test_ground_list_pred = y
                t_test_labels_list = labels_attack
                # X_aligned_within_class_list = X_aligned_within_class.data.cpu().numpy()
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_reconstructed_list = torch.cat((t_test_reconstructed_list, reconstructed), dim=0)
                t_test_ground_list_recons = torch.cat((t_test_ground_list_recons, x), dim=0)
                t_test_ground_list_pred = torch.cat((t_test_ground_list_pred, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels_attack), dim=0)
                # X_aligned_within_class_list = np.concatenate((X_aligned_within_class_list, X_aligned_within_class.data.cpu().numpy()), axis=0)
        
        test_loss_list.append(loss.item())
        acu_loss += loss.item()
        
        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

        avg_loss = sum(test_loss_list)/len(test_loss_list)

    test_reconstructed_list = t_test_reconstructed_list.tolist()
    test_predicted_list = t_test_predicted_list.cpu().detach().numpy()      
    test_ground_list_recons = t_test_ground_list_recons.tolist()     
    test_ground_list_pred = t_test_ground_list_pred.cpu().detach().numpy()      
    test_labels_list = t_test_labels_list.tolist()   
    # X_aligned_within_class_1 = X_aligned_within_class_list
    # if mode=="test":
    #     plot_mean_signal(X_aligned_within_class_1, np.array(test_ground_list_recons), ratio=[10,4],
    #                              class_num=0)
    return avg_loss, test_reconstructed_list, test_predicted_list, test_ground_list_recons, test_ground_list_pred, test_labels_list




