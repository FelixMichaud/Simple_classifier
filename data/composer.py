#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 20:25:39 2020

@author: felix
"""
from glob import glob
from itertools import permutations
import random
import librosa
import numpy as np
from Management import Management


def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def stretch(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)


def augmentation(audio_to_augment):
    n = random.randint(0, 1)
    if n == 0:
        t = random.randint(0, 1)
        if t == 1:
            factor_pitch = random.uniform(-1, 0.01)
            audio_to_augment = pitch(audio_to_augment, sr, factor_pitch)
        if t == 0:
            factor_stretch = random.uniform(0.9, 1.1)
            audio_to_augment = stretch(audio_to_augment, factor_stretch)
    else:
        for ii in range(n):
            factor_pitch = random.uniform(-1, 0.01)
            audio_to_augment = pitch(audio_to_augment, sr, factor_pitch)
            factor_stretch = random.uniform(0.9, 1.1)
            audio_to_augment = stretch(audio_to_augment, factor_stretch)
    return audio_to_augment




def composer(management, bird_folder, n_train):   
    """
    
    Parameters
    ----------
    management : Class
        management methods for audio 
    bird_folder : str
        folder names
    n_train : int
        nb of files to combine

    Returns
    -------
    Sets of data

    """
    audio_files = management.audio_number(bird_folder, n_train)
    combined_files = list(permutations(audio_files, 3))
    nb_audio = len(combined_files)  
    print(bird_folder, "bird_folder")
    print(nb_audio,"nb_audio")
    random.shuffle(combined_files)
    
    trainset_ind = round(nb_audio*0.8)
    validationset_ind = round(trainset_ind*0.05)
    
    trainset      = []
    validationset = []
    testset       = []
    
    trainset      = combined_files[:trainset_ind]
    validationset = trainset[:validationset_ind]
    testset       = combined_files[trainset_ind:] 
    return trainset, validationset, testset


def saver(management, audio_path, sr, dataset_path, n_files, list_files):                
            for count, trio in enumerate(list_files):
                final_audio = np.array([])
                if count > n_files:
                    break
                for file in trio:
                    audio = management.normalization(management.filt(management.load_file(file, sr), sr))
                    audio = audio - np.mean(audio)
                    audio = augmentation(audio)
                    final_audio = np.append(final_audio, audio)
                
                bird = management.bird_name(file)
                management.createFolder(audio_path + dataset_path + bird)
                management.save_audio(audio_path + dataset_path + bird 
                                      + "/" + bird 
                                      + str(count) + ".wav", final_audio, sr)



if __name__ == '__main__':
    # arguments

    sr = 16000
    path = "./audio"
    dataset_train_path = "/trainset/"
    dataset_validation_path = "/validationset/"
    dataset_test_path  = "/test_set/"
    bird_folders = glob(path + "/raw_data/*/")
    bird_folders.sort()
    management = Management()


    
    for bird in bird_folders:
        train, validation, test = composer(management, bird, n_train=18)
        saver(management, path, sr, dataset_train_path, n_files=len(train), list_files=train)
        saver(management, path, sr, dataset_validation_path, n_files=len(validation), list_files=validation)
        saver(management, path, sr, dataset_test_path, n_files=len(test), list_files=test)
