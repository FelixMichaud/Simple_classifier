#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 19:14:47 2020

@author: felix
"""
from glob import glob

import os
import fnmatch


def createFolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_list(path, ext):
    list_audios = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, '*' + ext):
            list_audios.append(os.path.join(root, filename))
    return list_audios


def num_char(x):
    return(x.rsplit("(", 1)[0].split("+", 1)[1]) 


path = "./audio"

list_folders = glob("./audio/*/")

for ii in range(len(list_folders)):
    audio_files = create_list(list_folders[ii], ".wav")
    for jj in range(len(audio_files)):
        os.rename(audio_files[jj], list_folders[ii] + num_char(audio_files[jj]) + str(jj) + ".wav")        



