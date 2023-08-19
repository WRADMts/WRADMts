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

class CombinedModel(nn.Module):
  def __init__(self, wtan, glm):
    super(CombinedModel, self).__init__()
    self.wtan = wtan
    self.glm = glm
    
  def forward(self, x, edge_index, mode = ''):
    # Apply the temporal transformation to the feature vectors
    x, thetas, gamma = self.wtan(x, return_theta=True)

    # Apply the GNN to the graph and feature vectors
    out_recons, out_pred, att_weight = self.glm(x, edge_index)

    return out_recons, out_pred, att_weight, thetas, x, gamma
