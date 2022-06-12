#i redid most of it without looking

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

#randomize weights
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)

#define model
def model(inputs):
    return inputs @ w.t() + b


#define loss (MSE)
def loss_fn(preds, targets=targets):
    diff = preds-targets
    return torch.sum(diff * diff)/diff.numel()


for i in range(epochs):
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    loss.backward()

    with torch.no_grad():
        w -= w.grad * alpha
        b -= b.grad * alpha

        w.grad.zero_()
        b.grad.zero_()


final_preds = model(inputs)
print(loss_fn(final_preds))
print(loss_fn(final_preds)**(1/2))