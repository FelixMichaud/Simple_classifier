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


  
def load_data(management, path, rect_win, nwin, nstep, sr):
    bird_folders = glob(path+"*/")
    bird_folders.sort()
    partition = dict()  
    label = dict()  
    
    list_name = [] 
    list_label = []
    for count, bird in enumerate(bird_folders):
        list_calls = management.create_list(bird, ext)
        for audio in list_calls:   
            calls = management.load_file(audio, sr)
            Mat = trames(calls, window=rect_win, nwin=nwin, nstep=nstep, fs=sr)
            for frames in Mat:
                print(frames, 'frames')
                list_name.append(frames)
                print(count, "count")
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
    
    
    train_partition, train_label = load_data(management, path+dataset_train_path
                                             , rect_win, 
                                             nwin, 
                                             nstep, 
                                             sr
                                             )




with open('./train_partition.pickle', 'wb') as handle:
    pickle.dump(train_partition, handle, protocol=pickle.HIGHEST_PROTOCOL)


with open('./train_label.pickle', 'wb') as handle:
    pickle.dump(train_label, handle, protocol=pickle.HIGHEST_PROTOCOL)



    

 











