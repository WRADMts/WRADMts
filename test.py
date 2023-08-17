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

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels_attack, labels_recons, edge_index in dataloader:
        x, y, labels_attack, labels_recons, edge_index = [item.to(device).float() for item in [x, y, labels_attack, labels_recons, edge_index]]
        
        with torch.no_grad():
            ## getting prediction and reconstruction output on test data
            reconstructed, predicted, att_weight, thetas, X_aligned_within_class, gamma = model(x, edge_index, mode) 

            loss_recons = loss_func(reconstructed.float().to(device), x)
            loss_pred = loss_func(predicted.float().to(device), y)
            loss = 1*loss_pred+1*loss_recons      

            labels_attack = labels_attack.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_reconstructed_list = reconstructed
                t_test_ground_list_recons = x
                t_test_ground_list_pred = y
                t_test_labels_list = labels_attack
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_reconstructed_list = torch.cat((t_test_reconstructed_list, reconstructed), dim=0)
                t_test_ground_list_recons = torch.cat((t_test_ground_list_recons, x), dim=0)
                t_test_ground_list_pred = torch.cat((t_test_ground_list_pred, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels_attack), dim=0)
        
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
    return avg_loss, test_reconstructed_list, test_predicted_list, test_ground_list_recons, test_ground_list_pred, test_labels_list




