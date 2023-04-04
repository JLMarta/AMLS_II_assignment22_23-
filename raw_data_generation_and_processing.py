from scipy.io.wavfile import read
import numpy as np
import math
from tqdm import tqdm
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def read_all_data(path):
    df = pd.read_csv('./raw_dataset/REFERENCE123456.csv', names = ['filename','label'])
    seg_labels = []
    seg_sigs = []
    for i in tqdm(range(0,len(df))):
        filename = df.iloc[i,0]
        fullname = path + filename + '.wav'
        fs, sig = read(fullname)
        max_seg = math.floor(len(sig)/(2*fs))
        label = df.iloc[i,1]
        for j in range(0,max_seg):
                seg_sig = sig[j*2*fs:(j+1)*2*fs]
                seg_label = label
                seg_sigs.append(seg_sig)
                seg_labels.append(seg_label)
    seg_sigs = np.array(seg_sigs)
    seg_labels = np.array(seg_labels)
    return seg_sigs, seg_labels

def data_split(X,y):
    X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
    print(X_train.shape), print(y_train.shape)
    print(X_valid.shape), print(y_valid.shape)
    print(X_test.shape), print(y_test.shape)
    return X_train, y_train, X_valid, y_valid, X_test, y_test

def plot_raw_image(seg_sigs, seg_labels, path):
    for i in tqdm(range(0,len(seg_sigs))):
        sig = seg_sigs[i,:]
        label = seg_labels[i]
        dstdir = path
        isExist = os.path.exists(dstdir)
        if not isExist:
            os.makedirs(dstdir)
        if label == 1:
            subdir = '/Abnormal'
            dir = dstdir + subdir
            isExist = os.path.exists(dir)
            if not isExist:
                os.makedirs(dir)
            filepath = dir + ('/Sig') + str(i) + ('.jpg')
            plt.figure(figsize=(200/96, 200/96), dpi=96)
            plt.plot(sig)
            plt.axis('off')
            plt.savefig(filepath)
        elif label == -1:
            subdir = '/Normal'
            dir = dstdir + subdir
            isExist = os.path.exists(dir)
            if not isExist:
                os.makedirs(dir)
            filepath = dir + ('/Sig') + str(i) + ('.jpg')
            plt.figure(figsize=(200/96, 200/96), dpi=96)
            plt.plot(sig)
            plt.axis('off')
            plt.savefig(filepath)
        else:
            print('Unknown Error Happened')
