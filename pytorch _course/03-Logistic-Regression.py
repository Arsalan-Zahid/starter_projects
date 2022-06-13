from os import path
import torch
from torchvision.datasets import MNIST
import torchvision.transforms  as transforms
import numpy as np

epochs = 40
alpha = 0.0005

#get paths
current_path = path.dirname(__file__)
MNIST_path = path.join(current_path, 'MNIST_data')

#get the dataset
dataset = MNIST(root=MNIST_path, download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root=MNIST_path, train=False)




N = len(dataset)
#randomly split the dataset - CHECK LATER
from torch.utils.data import random_split
train_ds, val_ds = random_split(dataset, [int(N*0.8), int(N*0.2)])

from torch.utils.data import DataLoader
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)


#SETUP THE MODEL

import torch.nn as nn

input_size = 28**2
num_classes=10


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, num_classes)
    
    def forward(self, xb):
            xb = xb.reshape(-1, 784)
            out = self.linear(xb)
            return out

    def training_step(self, batch):
        images, labels = batch
        preds = self(images)
        loss = cross_entropy(preds, labels)
        return loss
        
    def val_step(self, batch):
        'a single validation step; returns its loss and acc'

        images, labels = batch
        preds = self(images)
        loss = cross_entropy(preds, labels)
        acc = accuracy(preds, labels)

        return {'val_loss': loss, 'val_acc': acc}

    def val_epoch_end(self, val_step_outputs):
        'compiles the '

        batch_losses = [x['val_loss'] for x in val_step_outputs]
        epoch_loss = torch.stack(batch_losses).mean()

        batch_acc = [x['val_acc'] for x in val_step_outputs]
        epoch_acc = torch.stack(batch_acc).mean()

        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

            
model = MnistModel()

print(model.linear.weight.shape)
print(model.linear.bias.shape)

import torch.nn.functional as F


#_______________________________________________________

def accuracy(outputs, labels):
    max_probs, preds = torch.max(outputs, dim=1)
    return (torch.sum(preds==labels)/len(labels))


def evaluate(model=model, val_loader=val_loader):
    outputs = [model.val_step(batch) for batch in val_loader]
    return model.val_epoch_end(outputs)


from torch.nn.functional import cross_entropy
from torch.optim.sgd import SGD
opt = SGD(model.parameters(), lr=alpha)




def train(epochs=epochs, model=model, train_loader=train_loader, val_loader=val_loader):

    for epoch in range(epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            opt.step()
            opt.zero_grad()

        eval_data = evaluate()
        print(epoch, 'val_loss, val_acc', eval_data['val_loss'], eval_data['val_acc'])
        
        

        

train()
