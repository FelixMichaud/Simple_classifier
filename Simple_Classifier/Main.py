#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 18:48:49 2020

@author: felix
"""


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



def accuracy(predicted, labels):
    class_predicted = torch.max(predicted, 1)[1] 
    correct = 0
    total = 0
    total += labels.size(0)
    correct += (class_predicted == labels).sum().item()
    return 100*( correct / total)


def acc_per_class(outputs, labels, nb_class):   
    class_correct = list(1. for i in range(nb_class))
    class_total   = list(1. for i in range(nb_class))   
    results       = []
    _, predicted  = torch.max(outputs, 1)   
    c             = (predicted == labels).squeeze()
    
    for i in range(len(labels)):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1
    for i in range(nb_class):
         results = np.append(results, 100 * class_correct[i] / class_total[i])
    return results


def train(net, loader_train, optimizer, args):
    torch.set_grad_enabled(True)
    num_batch = 0 
    criterion = nn.CrossEntropyLoss()     
    for ii, batch_data in enumerate(loader_train):
        frames           = batch_data[0]
        labels           = batch_data[1]
        class_prediction = net(frames) 
        #check class_prediction along loop
        loss             = criterion(class_prediction, labels)
        acc         = accuracy(class_prediction, labels)
#        acc_class   = acc_per_class(class_prediction, labels, 25)
#        print(acc_class, 'acc_class')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()      
        
        if ii%3 == 0: 
    #writing of the Loss values and elapsed time for every batch
            batchtime = (time.time() - args.starting_training_time)/60 #minutes
            #Writing of the elapsed time and loss for every batch 
            with open("./losses/loss_train/loss_times_train.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([str(loss.detach().cpu().numpy()), acc ,batchtime])                      
##save the model and predicted images every args.save_per_batchs        
        if ii%args.save_per_batchs == 0: 
            torch.save({
                'num_batch': num_batch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
                'Saved_models/model4b_{}.pth.tar'.format(ii))  


def evaluation(net, loader_train, optimizer, args):
    torch.set_grad_enabled(False)
    criterion = nn.CrossEntropyLoss()     
    for ii, batch_data in enumerate(loader_train):
        
        frames           = batch_data[0]
        labels           = batch_data[1]
        class_prediction = net(frames)        
        loss             = criterion(class_prediction, labels)
        acc         = accuracy(class_prediction, labels)
             
        if ii%3 == 0:         
    #writing of the Loss values and elapsed time for every batch
            batchtime = (time.time() - args.starting_training_time)/60 #minutes
            #Writing of the elapsed time and loss for every batch 
            with open("./losses/loss_eval/loss_times_eval.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerow([str(loss.detach().cpu().numpy()), acc ,batchtime])                      
 


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
##################################################
    pickle_eval_list = './data/eval_partition.pickle'
    pickle_eval_label_list = './data/eval_label.pickle'
    
    list_eval = open(pickle_eval_list, 'rb')
    eval_label= open(pickle_eval_label_list, 'rb')
    
    dict_eval = pickle.load(list_eval)
    dict_eval_label = pickle.load(eval_label)
    
    list_eval = list(dict_eval["audio_name"])
    list_eval_label = list(dict_eval_label["labels"])
###################################################


    train_partition = Dataset(list_train, list_label, nb_class=25)
    
    loader_train  = torch.utils.data.DataLoader(
    train_partition,
    batch_size = 16,
    shuffle=True,
    drop_last=True,
    num_workers=2)   
    
        
    eval_partition  = Dataset(list_eval, list_eval_label, nb_class = 25)
    
    loader_eval  = torch.utils.data.DataLoader(
    eval_partition,
    batch_size = 16,
    shuffle=True,
    drop_last=True,
    num_workers=2)      
      
    
    classifier = Classifier()    
    nb_parameter = count_parameters(classifier)        
    optimizer = create_optimizer(classifier, args)    
    args.starting_training_time = time.time() 
    
    if args.mode == 'train':
        
        #OverWrite the Files for loss saving and time saving
        fichierLoss = open("./losses/loss_train/loss_times_train.csv", "w")
        fichierLoss.close()
        
        fichierLoss = open("./losses/loss_eval/loss_times_eval.csv", "w")
        fichierLoss.close()       
        
        for epoch in range(0, 20):
            train(classifier, loader_train, optimizer, args) 
            if epoch%2==0:
                evaluation(classifier, loader_eval, optimizer, args)
    
    
    
    