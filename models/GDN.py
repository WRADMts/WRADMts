import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn import Parameter
import time
from util.time import *
from util.env import *
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F

from .graph_layer import GraphLayer


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, out_dim, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num-1:
                    modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, out_dim))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num ))
                modules.append(nn.BatchNorm1d(inter_num))
                # modules.append(nn.Tanh())
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x
        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1)
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()


        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, edge_index, weights, node_num=0):


        out = self.gnn(x, edge_index, weights)
  
        out = self.bn(out)
        
        return self.relu(out)
        # return self.tanh(out)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, a_init=1):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        self.weight_arr = Parameter(torch.Tensor(node_num, node_num))
        # self.weight_arr = torch.ones(node_num, node_num).to(device)                                                   

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim+embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.dim=dim
        self.input_dim=input_dim
        self.node_embedding = None
        self.a_init = a_init
        self.learned_graph = None

        self.out_layer_recons = OutLayer(dim*edge_set_num, node_num, out_layer_num, out_dim = input_dim, inter_num = out_layer_inter_dim)
        self.out_layer_pred = OutLayer(dim*edge_set_num, node_num, out_layer_num, out_dim = 1, inter_num = out_layer_inter_dim)
        
        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()
    
    def init_params(self):
        # nn.init.normal_(self.weight_arr, mean=0, std=2)
        # nn.init.ones_(self.weight_arr)
        nn.init.constant_(self.weight_arr, self.a_init)


    def forward(self, data, org_edge_index):

        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):

            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num*batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)
            
            batch_edge_index = self.cache_edge_index_sets[i]

            # print(batch_edge_index)

            gcn_out = self.gnn_layers[i](x, batch_edge_index, weights=self.weight_arr.repeat(batch_num, 1), node_num=node_num*batch_num)
           
            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        # out = x.permute(0,2,1)                                      ##new
        # out = F.relu(self.bn_outlayer_in(out))
        # out = out.permute(0,2,1)

        # out = self.dp(out)
        out_recons = self.out_layer_recons(x)                             #whether fnn is needed or not
        out_recons = out_recons.view(-1, node_num, self.input_dim)
        out_pred = self.out_layer_pred(x) 
        out_pred = out_pred.view(-1, node_num)##Pred
   
        return out_recons, out_pred, self.weight_arr
        