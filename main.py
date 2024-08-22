import os
import torch
import numpy as np
import dataset as ds
import torch.utils.data as data
from torch import nn, functional as F
from functools import partial

if __name__ == "__main__":
    dataset = partial(ds.ReverseDataset, 10, 32)
    train_loader = data.DataLoader(dataset(100000), batch_size=128,
                                   shuffle=True, drop_last=True,
                                   pin_memory=True)
    val_loader   = data.DataLoader(dataset(5000), batch_size=128)
    test_loader  = data.DataLoader(dataset(20000), batch_size=128)

    inputs, labels = train_loader.dataset[0]
    print(f"Input data: {inputs}")
    print(f"Labels: {labels}")