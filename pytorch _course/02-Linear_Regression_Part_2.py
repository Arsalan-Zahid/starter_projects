import numpy as np
import torch

#hyperparameters
batch_size = 5

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')


targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')


#load
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#load TensorDataset
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

#initialize weights and biases
model = torch.nn.Linear(3, 2)

from torch.nn.functional import mse_loss


#define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

def fit(train_dl, epochs, model=model, loss_fn=mse_loss, opt=opt):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            preds = model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()

            #take a step
            opt.step()
            opt.zero_grad()


fit(train_dl, 150)
final_preds = model(inputs)
print(mse_loss(final_preds,targets))
print(mse_loss(final_preds,targets)**(1/2))
