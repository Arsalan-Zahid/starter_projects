import numpy as np
import torch

alpha = 1e-5
epochs = 150


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


def model(x):
    return x @ w.t() + b

def msqe(t1, t2):
    diff = t1-t2
    return torch.sum(diff * diff)/diff.numel()

preds = model(inputs)
loss = msqe(preds, targets)

#get gradients
loss.backward()

#gradients are stored in the .grad property of their tensor



for i in range(epochs):

    #compute the loss, then the gradients
    #you actually have to repredict every time
    preds = model(inputs)
    loss = msqe(preds, targets)
    loss.backward()

    #no grad because we don't want to change the gradient while we're running it
    with torch.no_grad():
            
            #adjust the weights and biases, taking a small step from the gradient.
            w -= w.grad * alpha
            b -= b.grad * alpha

            #reset the gradients (we'll compute them later inshallah)
            w.grad.zero_()
            b.grad.zero_()



#now check out the msqe
preds = model(inputs)
msqe_final = (msqe(preds, targets))
from math import sqrt
print(sqrt(msqe_final))