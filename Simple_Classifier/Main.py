#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:48:49 2020

@author: felix
"""


from Management import Management
import numpy as np
from arguments import ArgParser
import csv
from dataset import Dataset
import pickle 
import torch 
import torch.nn as nn
from Modelnet import Classifier
import time


def create_optimizer(nets, args):
    net_sound = nets
    param_groups = [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    return torch.optim.Adam(param_groups)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(net, loader_train, optimizer, args):
    torch.set_grad_enabled(True)
    num_batch = 0 
    criterion = nn.CrossEntropyLoss()     
    for ii, batch_data in enumerate(loader_train):
        
        frames           = batch_data[0]
        labels           = batch_data[1]
        print(np.size(labels), 'shape labels')
        class_prediction = net(frames)        
        loss             = criterion(class_prediction, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
        
#writing of the Loss values and elapsed time for every batch
        batchtime = (time.time() - args.starting_training_time)/60 #minutes
        #Writing of the elapsed time and loss for every batch 
        with open("./losses/loss_train/loss_times_train.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([str(loss.detach().cpu().numpy()), batchtime])                      
##save the model and predicted images every args.save_per_batchs        
        if ii%args.save_per_batchs == 0: 
            torch.save({
                'num_batch': num_batch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                'Saved_models/model4b_{}.pth.tar'.format(ii))  




if __name__ == '__main__':
    # arguments
    parser = ArgParser()
    args = parser.parse_train_arguments()
    #protocol=pickle.HIGHEST_PROTOCOL
    pickle_train_list = './data/train_partition.pickle'
    pickle_label_list = './data/train_label.pickle'
    
    list_train = open(pickle_train_list, 'rb')
    list_label= open(pickle_label_list, 'rb')
    
    dict_train = pickle.load(list_train)
    dict_label = pickle.load(list_label)
    
    list_train = list(dict_train["audio_name"])
    list_label = list(dict_label["labels"])


    train_partition = Dataset(list_train, list_label, nb_class=25)
        
    loader_train  = torch.utils.data.DataLoader(
    train_partition,
    batch_size = 2,
    shuffle=True,
    num_workers=2)   
    
    classifier = Classifier()
    
    nb_parameter = count_parameters(classifier)
        
    optimizer = create_optimizer(classifier, args)
    
    args.starting_training_time = time.time() 
    
    if args.mode == 'train':
        for epoch in range(0, 2):
            train(classifier, loader_train, optimizer, args) 
    
    
    
    