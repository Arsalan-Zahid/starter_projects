#i redid most of it without looking

import numpy as np
import torch

alpha = 1e-5
epochs = 200


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')

#convert to pytorch tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

#initialize random weights
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

def msqe(preds, targets):
    diff = preds-targets
    return torch.sum(diff * diff)/diff.numel()


def model(inputs):
    return inputs @ w.t() + b



#train the model

for i in range(epochs):
    #compute preds, loss, then gradients
    preds = model(inputs)
    loss = msqe(preds, targets)
    loss.backward()

    #without changing the gradients, update the weights
    with torch.no_grad():
        w -= w.grad * alpha
        b -= b.grad * alpha
        w.grad.zero_()
        b.grad.zero_()


#get preds and final_msqe

final_preds = model(inputs)
final_msqe = msqe(final_preds, targets)
from math import sqrt

print(sqrt(final_msqe))