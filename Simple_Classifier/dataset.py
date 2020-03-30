#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:49:04 2020

@author: felix
"""

#import torch
from torch.utils import data
import torch
import numpy as np

class Dataset(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, labels, nb_class):
        'Initialization'
        self.labels   = labels
        self.list_IDs = list_IDs
        self.nb_class = nb_class
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        audio_data = torch.FloatTensor(self.list_IDs[index])
        label = self.labels[index]
        # vector_label = np.zeros((self.nb_class), dtype=np.int64)
        # vector_label[label] = 1

        return audio_data, label