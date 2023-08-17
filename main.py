# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score, precision_recall_curve 

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset


from models.GLM import GLM
from models.train_utils import WTAN_args
from WTAN.WTAN_layer import WTAN
from wtan_gdn import CombinedModel


from train import train
from test  import test
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores, get_scores_topk

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset'] 

        ## read input

        train_orig = pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_orig = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)   
        train, test = train_orig, test_orig  

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        ## build graph edges

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype = torch.long)

        self.feature_map = feature_map

        self.labels = test.attack.tolist()

        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=self.labels)


        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0, worker_init_fn=np.random.seed(train_config['seed']))

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        # Arguments for WTAN network with novel alignment loss
        # WTAN args with smoothness prior
        self.WTANargs1 = WTAN_args(tess_size=32,
                              smoothness_prior=True,
                              lambda_smooth=1,
                              lambda_var=0.1,
                              n_recurrences=1,
                              zero_boundary=True,
                              )

        ## alignment network instance
        channels, input_shape = self.train_dataloader.dataset[0][0].shape
        self.wtan = WTAN(input_shape, channels, tess=[self.WTANargs1.tess_size,], n_recurrence=self.WTANargs1.n_recurrences,
                    zero_boundary=self.WTANargs1.zero_boundary, device='gpu').to(self.device)

        self.WTANargs1.T = self.wtan.get_basis()

        ##Graph learning network instance

        self.glm = GLM(edge_index_sets, len(feature_map),
                dim=train_config['dim'], 
                input_dim=train_config['slide_win'],
                out_layer_num=train_config['out_layer_num'],
                out_layer_inter_dim=train_config['out_layer_inter_dim'],
                a_init=train_config['a_init'],   
            ).to(self.device)


        # Create an instance of combined model (alignment+graph learning)
        self.model = CombinedModel(self.wtan, self.glm)

    def run(self):

        print('===========**train**================')
        if len(self.env_config['load_model_path']) > 0:
            self.model_save_path = self.env_config['load_model_path']
        else:
            self.model_save_path = self.get_save_path()[0]

            self.train_log = train(self.WTANargs1, self.model, self.model_save_path,
                config = self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config['dataset'],
                gt_labels = self.labels
            )

    def test(self):

        print('=========================** test **============================')

        # # test            
        self.model.load_state_dict(torch.load(self.model_save_path))
        best_model = self.model.to(self.device)

        ## computing anomaly scores
        _, self.test_result_recons, self.test_result_pred, self.test_ground_recons, self.test_ground_pred, test_labels = test(best_model, self.test_dataloader, 'test')
        test_scores_recons, test_scores_pred = self.get_score(self.test_result_recons, self.test_result_pred, self.test_ground_recons, self.test_ground_pred, test_labels)
        
        top_recons = get_scores_topk(test_scores_recons, topk=self.train_config['topk'])
        top_pred = get_scores_topk(test_scores_pred, topk=self.train_config['topk'])
        ## normalizing anomaly scores
        normalized_top_recons = (top_recons-np.min(top_recons))/(np.max(top_recons)-np.min(top_recons))
        normalized_top_pred = (top_pred-np.min(top_pred))/(np.max(top_pred)-np.min(top_pred))

        normalized_top_recons = np.append(normalized_top_recons, 0)
        normalized_top_pred = np.append(np.zeros(train_config['slide_win']),normalized_top_pred)

        top_recons = np.append(top_recons, 0)
        top_pred = np.append(np.zeros(train_config['slide_win']),top_pred)

        test_scores1 = top_pred + top_recons
        ## combined anomaly score from both reconstruction and prediction
        test_scores = normalized_top_pred + normalized_top_recons

        top1_best_info = get_best_performance_data(test_scores, self.labels)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info

        print(f'F1 score point adj: {info[0]}')
        print(f'F1 score : {info[1]}')
        print(f'prec_bfr : {info[2]}')
        print(f'rec_bfr: {info[3]}')
        print(f'precision: {info[4]}')
        print(f'recall: {info[5]}\n')
        print(f'aucroc:{info[6]}')

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        random.seed(seed)
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True, worker_init_fn=np.random.seed(seed))

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False, worker_init_fn=np.random.seed(seed))

        return train_dataloader, val_dataloader

    def get_score(self, test_result_recons, test_result_pred, test_ground_recons, test_ground_pred, test_labels):
        test_scores_recons, test_scores_pred = get_full_err_scores(test_result_recons, test_result_pred, test_ground_recons, test_ground_pred, test_labels)
        return test_scores_recons, test_scores_pred


    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']
        
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr          

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type = int, default=128)
    parser.add_argument('-epoch', help='train epoch', type = int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type = int, default=15)
    parser.add_argument('-dim', help='dimension', type = int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type = int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type = str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type = str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type = str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type = int, default=0)
    parser.add_argument('-comment', help='experiment comment', type = str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type = int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type = int, default=256)
    parser.add_argument('-decay', help='decay', type = float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type = float, default=0.1)
    parser.add_argument('-topk', help='topk num', type = int, default=20)
    parser.add_argument('-report', help='best / val', type = str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type = str, default='')
    parser.add_argument('-lamda', help='norm loss weight', type = float, default=0.00001)
    parser.add_argument('-a_init', help='adjacency weight initialization', type = int, default=1)
    parser.add_argument('-weight_p', help='pred loss weight', type = int, default=1)
    parser.add_argument('-weight_r', help='recons loss weight', type = int, default=1)
    parser.add_argument('-train_split', help='train test split', type = float, default=0.8)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'lamda': args.lamda,
        'a_init': args.lamda,
        'weight_p': args.weight_p,
        'weight_r': args.weight_r,
        'train_split': args.train_split,
    }

    env_config={
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }
    

    main = Main(train_config, env_config, debug=False)
    main.run()
    main.test()





