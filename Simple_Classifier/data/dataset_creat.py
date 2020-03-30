#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 16:24:59 2020

@author: felix
"""

from glob import glob
from Management import Management
import numpy as np
import pickle

'''
Save the 4000/0.25s audio segments in a pickle file with label

'''

def trames(audio, window, nwin, nstep, fs):
    # hop_size in ms   
    frame_num = int((len(audio) - nwin) / nstep) + 1
    #rows   = nb trame
    #column = temp infoomation 
    list_segments = []    
    for n in range(frame_num):
        list_segments.append([window * audio[int(n*nstep):int(n*nstep+nwin)]])
    
    return list_segments


  
def load_data(management, nb_class, path, rect_win, nwin, nstep, sr):
    bird_folders = glob(path+"*/")
    bird_folders.sort()
    print(len(bird_folders), 'number of classes')
    partition = dict()  
    label = dict()  
    
    list_name = [] 
    list_label = []
    for count, bird in enumerate(bird_folders):
        if count > nb_class:
            break
        list_calls = management.create_list(bird, ext)
        for audio in list_calls:   
            calls = management.load_file(audio, sr)
            Mat = trames(calls, window=rect_win, nwin=nwin, nstep=nstep, fs=sr)
            for frames in Mat:
                list_name.append(frames)
                list_label.append(count)
    partition["audio_name"] = list_name
    label["labels"] = list_label
    return partition, label



if __name__ == '__main__':
    # arguments


    path = "./audio"
    dataset_train_path = "/trainset/"
    dataset_validation_path = "/validationset/"
    dataset_test_path  = "/test_set/"
    management = Management()
    ext = '.wav' 
    sr = 16000   
    
    
    win = 0.25
    step = win*(1-0.75)
    nwin = int(win*sr) 
    nstep = sr * step
    ## nb de points dans 1 trame
    rect_win = np.ones((nwin))
    
    
    eval_partition, eval_label = load_data(management,
                                             25,
                                             path+dataset_validation_path,
                                             rect_win, 
                                             nwin, 
                                             nstep, 
                                             sr
                                             )




with open('./eval_partition.pickle', 'wb') as handle:
    pickle.dump(eval_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./eval_label.pickle', 'wb') as handle:
    pickle.dump(eval_label, handle, protocol=pickle.HIGHEST_PROTOCOL)



    

 











