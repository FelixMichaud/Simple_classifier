#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 18:32:28 2020

@author: felix
"""
import torch.nn as nn
import numpy as np


class Classifier(nn.Module):

    def __init__(self, frame_dim: int = 4000, n_classes: int = 25):
        self.frame_dim = frame_dim
        self.n_classes = n_classes      
    
    
        super(Classifier, self).__init__()
        self.ConvL1  = nn.Conv1d(in_channels = 1,
	                    out_channels =16,
	                    kernel_size = 32,
	                    stride = 2
	                    )
        self.norm1    = nn.BatchNorm1d(16)
        self.Pool1   = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ConvL2  = nn.Conv1d(in_channels = 16,
	                    out_channels =32,
	                    kernel_size = 16,
	                    stride = 2
	                   )
        self.norm2    = nn.BatchNorm1d(32)
        self.Pool2   = nn.MaxPool1d(kernel_size=2, stride=2)
        self.ConvL3  = nn.Conv1d(in_channels = 32,
	                    out_channels =64,
	                    kernel_size = 8,
	                    stride = 2
	                    )
    
        self.norm3    = nn.BatchNorm1d(64)
        self.linear1 = nn.Linear(in_features=119, out_features=64)
        self.linear2 = nn.Linear(in_features=64, out_features=25)
        self.softmax = nn.Softmax()
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)


    def forward(self, x):
        x1 = self.relu(self.ConvL1(x))
        x1 = self.norm1(x1)
        x2 = self.relu(self.Pool1(x1))
        x3 = self.relu(self.ConvL2(x2))
        x3 = self.norm2(x3)
        x4 = self.relu(self.Pool2(x3))
        x5 = self.relu(self.ConvL3(x4))
        x5 = self.norm3(x5)
        x6 = self.dropout(self.relu(self.linear1(x5)))
        x7 = self.dropout((self.linear2(x6)))
        #classes = self.softmax(x7)
        return x7
