#plot
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random


# train_orig = pd.read_csv(f'./data/hai/train.csv', sep=',', index_col=0)
# test_orig = pd.read_csv(f'./data/hai/test.csv', sep=',', index_col=0)      
# train, test = train_orig.values, test_orig.values

# anom_scores = pd.read_csv(f'/home/abilasha/Dropbox/mts_ad_results/result_score/as_hai.csv', sep=',', index_col=False)

# min_max_scaler = MinMaxScaler()
# ans = min_max_scaler.fit_transform(train)

# min_max_scaler = MinMaxScaler()
# test_plot = min_max_scaler.fit_transform(test)

# min_max_scaler = MinMaxScaler()
# test_plot = min_max_scaler.fit_transform(anom_scores)

# fig, axs = plt.subplots(10)
# for i in range(10):
# 	axs[i].plot(test_plot[40000:40500,i], color = 'black')
# 	# axs[0].set_ylabel('Ourmodel' )

# # axs[8].plot(ans[:,0], color = 'blue')
# # axs[9].plot(test[:,-1], color = 'red')

# fig.tight_layout(pad=0.5)
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# plt.show()


# fig, axs = plt.subplots(8)
# axs[0].plot(test_plot[:,0], color = 'black')
# axs[0].set_ylabel('WRADMts' )
# axs[1].plot(test_plot[:,1], color = 'black')
# axs[1].set_ylabel('GDN' )
# axs[2].plot(test_plot[:,2], color = 'black')
# axs[2].set_ylabel('USAD' )
# axs[3].plot(test_plot[:,3], color = 'black')
# axs[3].set_ylabel('MTADGAT' )
# axs[4].plot(test_plot[:,4], color = 'black')
# axs[4].set_ylabel('Omni' )
# axs[5].plot(test_plot[:,5], color = 'black')
# axs[5].set_ylabel('Anom' )
# axs[6].plot(test_plot[:,6], color = 'black')
# axs[6].set_ylabel('Inter' )
# axs[7].plot(test_plot[:,7], color = 'black')
# axs[7].set_ylabel('MTGFlow' )
# [axs1.axvspan(100,200,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(750,850,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(6500,6700,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(6850,6995,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(7220,7380,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(7684,7788,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(8500,8845,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(8953,9158,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(9420,9533,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(15000,15104,facecolor="r",alpha=0.3) for axs1 in axs]
# # axs[7].plot(test[:,-1], color = 'red')
# # axs[7].set_ylabel('GT' )
# fig.tight_layout(pad=0.5)
# plt.show()

# fig, axs = plt.subplots(9, figsize=(10, 10))
# # ind=np.array([2,4,5,6,7,8,9,10])
# # ind=np.array([11,12,14,16,18,20,22,23])
# # ind=np.array([25,27,28,29,30,36,38,40])
# ind=np.array([5,8,11,18,23,27,28,36,38])
# for i in range(9):
# 	axs[i].plot(test_plot[:,ind[i]], color = 'black', linewidth=1)
# 	# axs[0].set_ylabel('Ourmodel' )

# # axs[8].plot(ans[:,0], color = 'blue')
# # axs[9].plot(test[:,-1], color = 'red')
# [axs1.axvspan(100,240,facecolor="y",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(750,900,facecolor="g",alpha=0.5) for axs1 in axs]
# [axs1.axvspan(6610,6750,facecolor="b",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(6900,7050,facecolor="b",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(7302,7405,facecolor="g",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(7704,7807,facecolor="g",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(8553,8840,facecolor="r",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(9100,9190,facecolor="c",alpha=0.4) for axs1 in axs]
# [axs1.axvspan(9450,9560,facecolor="m",alpha=0.3) for axs1 in axs]
# [axs1.axvspan(15100,15170,facecolor="m",alpha=0.3) for axs1 in axs]

# fig.supxlabel('Time Series', fontsize = 15)
# fig.supylabel('Metrcis', fontsize = 15)

# fig.tight_layout(pad=0.5)
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[]);
# plt.show()


test_orig = pd.read_csv(f'/home/abilasha/Dropbox/mts_ad_results/result_score/ecg_scores.csv', sep=',', index_col=0)
min_max_scaler = MinMaxScaler()
test_plot = min_max_scaler.fit_transform(test_orig)
test_plot = test_plot[:,:]

fig, axs = plt.subplots(9)
axs[0].plot(test_plot[:,0], color = 'red')
# axs[0].set_ylabel('TS' )
axs[1].plot(test_plot[:,3], color = 'black')
# axs[1].set_ylabel('WRADMts' )
axs[2].plot(test_plot[:,4], color = 'black')
# axs[2].set_ylabel('GDN' )
axs[3].plot(test_plot[:,5], color = 'black')
# axs[3].set_ylabel('USAD' )
axs[4].plot(test_plot[:,6], color = 'black')
# axs[4].set_ylabel('MTADGAT' )
axs[5].plot(test_plot[:,7], color = 'black')
# axs[5].set_ylabel('Omni' )
axs[6].plot(test_plot[:,8], color = 'black')
# axs[6].set_ylabel('Anom' )
axs[7].plot(test_plot[:,9], color = 'black')
# axs[7].set_ylabel('Inter' )
axs[8].plot(test_plot[:,10], color = 'black')
# axs[8].set_ylabel('MTGFlow' )
[axs1.axvspan(3900,4300,facecolor="y",alpha=0.3) for axs1 in axs]
[axs1.axvspan(700,1000,facecolor="r",alpha=0.3) for axs1 in axs]
[axs1.axvspan(7550,8100,facecolor="r",alpha=0.3) for axs1 in axs]
[axs1.axvspan(6768,7368,facecolor="r",alpha=0.3) for axs1 in axs]
fig.tight_layout(pad=0.1)
plt.show()