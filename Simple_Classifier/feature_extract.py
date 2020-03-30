#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 21:30:54 2020

@author: felix
"""

import os
import fnmatch
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import librosa
from spafe.features.mfcc import mfcc
from spafe.features.gfcc import gfcc


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
        
def create_list(path, ext):
    list_audios = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*' + ext):
            list_audios.append(os.path.join(root, filename))
    return list_audios


def bird_name(x):
    return x.rsplit(".", 1)[0].rsplit("/", 1)[1]  

def normalization(vector):
    return (vector-np.amin(vector))/(np.amax(vector)-np.amin(vector))

# init input vars
num_ceps = 12
low_freq = 20
high_freq = 8000
nfilts = 22
nfft = 512
dct_type = 2,
use_energy = False,
lifter = 5
normalize = True


list_folders = glob("data/audio/raw_data/*")
list_folders.sort()

for ii in range(len(list_folders)):
      audio_files = create_list(list_folders[ii], ".wav")
      audio_files.sort()
      for jj in range(len(audio_files)):     
         sig, fs = librosa.load(audio_files[jj],
                                sr=16000, 
                                mono=True,
                                dtype=float                       
                                )   
         sig = normalization(sig)      
         # compute features
         # mfccs = np.rot90(mfcc(sig=sig,                      
         #              fs=fs,
         #              num_ceps=num_ceps,
         #              nfilts=nfilts,
         #              nfft=nfft,
         #              low_freq=low_freq,
         #              high_freq=high_freq,
         #              dct_type=dct_type,
         #              use_energy=use_energy,
         #              lifter=lifter,
         #              normalize=normalize),1, (1, 0))
         
         
         # gfccs = np.rot90(gfcc(sig=sig,
         #     fs=fs,
         #     num_ceps=num_ceps,
         #     nfilts=nfilts,
         #     nfft=nfft,
         #     low_freq=low_freq,
         #     high_freq=high_freq,
         #     dct_type=dct_type,
         #     use_energy=use_energy,
         #     lifter=lifter,
         #     normalize=normalize),1, (1, 0))
         
    
         STFT = librosa.core.stft(sig, n_fft=1024, win_length=512)        
         # visualize spectogram
         fig1 = plt.figure(1)         
         plt.imshow(librosa.power_to_db(np.abs(STFT), ref=np.max), aspect='auto', origin='lower' )
         plt.colorbar()
         # visualize features
         # fig2 = plt.figure(2)
         # plt.imshow(mfccs, aspect='auto', origin='lower')
         # plt.colorbar()
         # visualize features
         # fig3 = plt.figure(3)
         # plt.imshow(gfccs, aspect='auto', origin='lower')
         # plt.colorbar()
        
         createFolder("./images/spectro/"+bird_name(list_folders[ii]))
         fig1.savefig("./images/spectro/"+bird_name(list_folders[ii]) +"/" + bird_name(audio_files[jj]), format="png")
#         createFolder("./images/mfcc/"+bird_name(list_folders[ii]))
         # fig2.savefig("./images/mfcc/"+bird_name(list_folders[ii]) +"/" + bird_name(list_folders[ii]) + str(jj),format="png")
         # createFolder("./images/gfcc/"+bird_name(list_folders[ii]))
         # fig3.savefig("./images/gfcc/"+bird_name(list_folders[ii]) +"/" + bird_name(list_folders[ii]) + str(jj),format="png")         
         plt.close("all")
        
      
        
    
       
        
        
        
        
        
        
        
        