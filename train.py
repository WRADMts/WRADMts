import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores, get_scores_topk
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
from WTAN.smoothness_prior import smoothness_norm
from WTAN.alignment_loss import alignment_loss
from torch.nn import Parameter


## computing loss value

def loss_func(i, y_pred_recons, y_true_recons, y_pred_pred, y_true_pred, att_weight, lamda, weight, thetas, WTANargs, channels, gamma):
    node_num=att_weight.shape[0]

    if (i<15):
        mse_loss_pred = F.mse_loss(y_pred_pred, y_true_pred, reduction='mean')
        mse_loss_recons = F.mse_loss(y_pred_recons, y_true_recons, reduction='mean')
    else:
        mse_loss_pred = F.l1_loss(y_pred_pred, y_true_pred, reduction='mean')
        mse_loss_recons = F.l1_loss(y_pred_recons, y_true_recons, reduction='mean')

    ## L1 regularization on adjacency matrix

    lambda_i = torch.Tensor(lamda*np.array([0.1]*node_num)).cuda()
    norm_loss = torch.sum(lambda_i*torch.norm(att_weight, p=1, dim=1))

    ##alignment loss computation

    align_loss = alignment_loss(y_true_recons, thetas, channels, WTANargs, gamma)

    loss = weight[0]*mse_loss_pred + weight[0]*mse_loss_recons+ norm_loss+ 0.01*align_loss
    return  loss, mse_loss_recons, mse_loss_pred, align_loss

def get_score(test_result_recons, test_result_pred, test_ground_recons, test_ground_pred, test_labels):
        test_scores_recons, test_scores_pred = get_full_err_scores(test_result_recons, test_result_pred, test_ground_recons, test_ground_pred, test_labels)
        return test_scores_recons, test_scores_pred


def train(WTANargs, model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None,  feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None, gt_labels = []):

    seed = config['seed']
    device = get_device()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    acu_loss = 0
    min_loss = 1e+8
    max_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    lamda = config['lamda']
    weight = np.array([config['weight_p'],config['weight_r']])

    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    channels, input_shape = train_dataloader.dataset[0][0].shape

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        acu_recons_loss = 0
        acu_pred_loss = 0
        acu_align_loss = 0
        model.train()

        for x, labels, attack_labels, recons_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()

            ## obtaining reocnstructed and predicted value for train data
            out_recons, out_pred, att_weight, thetas, x_aligned, gamma = model(x, edge_index)                                                 
            ## computing loss value for joint loss function
            loss, recons_loss, pred_loss, align_loss = loss_func(i_epoch, out_recons.float().to(device), x_aligned.float().to(device), out_pred.float().to(device), labels, att_weight, lamda, weight, thetas, WTANargs, channels, gamma)
            
            loss.backward()
            optimizer.step()

            train_loss_list.append(loss.item())
            acu_loss += loss.item()
            acu_recons_loss += recons_loss.item()
            acu_pred_loss += pred_loss.item()
            acu_align_loss += align_loss.item()
                
            i += 1


        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f}), recons_loss:{:.8f}, pred_loss:{:.8f}, align_loss:{:.8f}'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss, acu_recons_loss/len(dataloader), acu_pred_loss/len(dataloader), acu_align_loss/len(dataloader)), flush=True
            )

        # use val dataset to judge for saving best model and early stopping
        if val_dataloader is not None:

            val_loss, val_reconstructed_list, val_predicted_list, _, _, _ = test(model, val_dataloader, 'val')

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss

    return train_loss_list
