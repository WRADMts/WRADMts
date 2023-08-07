import torch
import torch.nn as nn
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1,**kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weights = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        zeros(self.bias)


    def forward(self, x, edge_index, weights):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)                                       
            x = (x, x)
        else:
            x = (x[0], x[1])
            x = (self.lin(x[0]), self.lin(x[1]))

        # edge_index, _ = remove_self_loops(edge_index)                                                         ##add and remove self loop
        # edge_index, _ = add_self_loops(edge_index,
        #                                num_nodes=x.size(self.node_dim))

        out = self.propagate(edge_index, x=x, weights=weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        # return out, self.weights
        return out


    def message(self, x_j, weights):

        # print(edge_index_j)
        # print(x_j.shape)

        x_j = x_j.view(-1, self.heads, self.out_channels)
        # print(x_j.shape)
        # weights=F.dropout(weights, self.dropout, self.training)
        # self.weights = weights

        # weights = weights.view(-1, self.heads, 1)

        # return x_j
        return x_j * weights.view(-1, self.heads, 1)


    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
