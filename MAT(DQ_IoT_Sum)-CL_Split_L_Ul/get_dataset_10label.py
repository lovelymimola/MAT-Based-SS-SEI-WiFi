import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(ft, random_num):
    x = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_X_train.npy")
    y = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_Y_train.npy")
    y = y.astype(np.uint8)
    X_train_labeled1, X_train_unlabeled1, Y_train_labeled1, Y_train_unlabeled1 = train_test_split(x, y, test_size=0.5, random_state=random_num)
    X_train_labeled2, X_train_unlabeled2, Y_train_labeled2, Y_train_unlabeled2 = train_test_split(X_train_labeled1,Y_train_labeled1, test_size=0.6, random_state=random_num)
    X_train_labeled3, X_train_unlabeled3, Y_train_labeled3, Y_train_unlabeled3 = train_test_split(X_train_labeled2,Y_train_labeled2,test_size=0.5,random_state=random_num)

    X_train_label, X_val, Y_train_label, Y_val = train_test_split(X_train_labeled3, Y_train_labeled3, test_size=0.3, random_state=random_num)

    X_train_unlabeled = np.concatenate((X_train_unlabeled1,X_train_unlabeled2,X_train_unlabeled3), axis=0)
    Y_train_unlabeled = np.concatenate((Y_train_unlabeled1, Y_train_unlabeled2,Y_train_unlabeled3), axis=0)

    X_train = np.concatenate((X_train_label, X_train_unlabeled), axis=0)
    Y_train = np.concatenate((Y_train_label, Y_train_unlabeled), axis=0)

    return X_train_label, X_train_unlabeled, X_val, Y_train_label, Y_train_unlabeled, Y_val

def TestDataset(ft):
    x = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_X_test.npy")
    y = np.load(f"/data/fuxue/WiFi/Dataset/Feet{ft}_Y_test.npy")
    y = y.astype(np.uint8)
    return x, y
