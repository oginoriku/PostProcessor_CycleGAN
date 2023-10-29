import numpy as np
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import loadPickle

def data_loader(data_path, clean_data_num, train_data_num, batch_size, val_data_num, val_batch_size, num_workers):
    # データセットのパスを定義
    clean_data_path = os.path.join(data_path, "clean_data")
    train_data_path = os.path.join(data_path, "train_data")
    val_data_path = os.path.join(data_path, "test_data")

    print("=====================================Clean Data Loading============================================")
    cleanset = []
    for i in tqdm(range(clean_data_num)):
        cleandata = loadPickle(os.path.join(clean_data_path, f"data{i}.pickle"))
        cleanset.append(cleandata)
    cleanset = np.array(cleanset)
    cleanloader = DataLoader(cleanset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    print("cleanset.shape: ", cleanset.shape)
    print("バッチサイズ: ", batch_size)
    print("イテレーション数: ", len(cleanloader))

    print("=====================================Train Data Loading============================================")
    trainset = []
    for i in tqdm(range(train_data_num)):
        traindata = loadPickle(os.path.join(train_data_path, f"data{i}.pickle"))
        trainset.append(traindata)
    trainset = np.array(trainset)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    print("trainset.shape: ", trainset.shape)
    print("バッチサイズ: ", batch_size)
    print("イテレーション数: ", len(trainloader))

    print("=====================================Val Data Loading============================================")
    valset = []
    for i in tqdm(range(val_data_num)):
        valdata = loadPickle(os.path.join(val_data_path, f"data{i}.pickle"))
        valset.append(valdata)
    valset = np.array(valset)
    valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    print("valset.shape: ", valset.shape)
    print("バッチサイズ: ", val_batch_size)
    print("イテレーション数: ", len(valloader))

    return cleanloader, trainloader, valloader

