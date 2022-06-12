from os import path
import torch
from torchvision.datasets import MNIST
import torchvision.transforms  as transforms
import numpy as np

#get paths
current_path = path.dirname(__file__)
MNIST_path = path.join(current_path, 'MNIST_data')

#get the dataset
dataset = MNIST(root=MNIST_path, download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root=MNIST_path, train=False)


def split_indicies(n, val_pct):
    #determine size of val
    n_val = int(n*val_pct)
    idxs = np.random.permutation(n)
    return idxs[n_val:], idxs[:n_val]


train_idxs, test_idxs = split_indicies(len(dataset), 0.2)
print(len(train_idxs), len(test_idxs))





import torch.nn as nn