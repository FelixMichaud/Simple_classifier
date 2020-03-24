#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 19:09:04 2020

@author: felix
"""

import os
import fnmatch
from scipy import signal
import numpy as np
import librosa

class Management():
        
    def createFolder(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
    def create_list(self, path, ext):
        list_audios = []
        for root, dirnames, filenames in os.walk(path):
            for filename in fnmatch.filter(filenames, '*' + ext):
                list_audios.append(os.path.join(root, filename))
                list_audios.sort()
        return list_audios
    
    
    def audio_number(self, nm_folders, nb):
        audio_files = self.create_list(nm_folders, ".wav")
        if len(audio_files) > nb:
            return audio_files[:nb]
    
    
    def load_file(self, file, sr):
        audio_raw, rate = librosa.load(file, sr=sr, mono=True)    
        return audio_raw
    
    
    def filt(self, audio_raw, rate):
        numtaps = 15     # Size of the FIR filter.    
        taps = signal.butter(numtaps, 800, 'highpass', fs=rate, output='sos')     
        sig_filt = signal.sosfilt(taps, audio_raw)
        return sig_filt
    
    
    def normalization(self, vector):
        return (vector-np.amin(vector))/(np.amax(vector)-np.amin(vector))
    
    
    def bird_name(self, x):
        return x.rsplit("/", 1)[0].rsplit("/", 1)[1]  
    
    
    
    def save_audio(self, path_to, audio_array, sr):
        librosa.output.write_wav(path_to, audio_array, sr, norm=False)
