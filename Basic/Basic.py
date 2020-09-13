# Loading necessary library
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms



#####################################
#      1. Basic autograd example
#####################################

x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

print(" x : {}    w : {}    b : {}".format(x,w,b))

# Building a computational graph
y = w * x + b
print("Value of y is :", y)

# Compute Gradients
y.backward()

# print out the gradient
print("x grad : {} w grad : {} b grad : {}".format(x.grad, w.grad, b.grad))



###########################################
#         2. Basic autograd example 2
###########################################

# Create tensor of shape (10, 3) and (10, 2)
x = torch.randn(10, 3)
y = torch.randn(10, 2)

# Building a fully connected layer
linear = nn.Linear(3, 2)
print('weight is :', linear.weight)
print('bias is :', linear.bias)

# Building loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)

# Forward pass
pred = linear(x)

# Compute loss
loss = criterion(pred, y)
print("loss is ", loss.item())

# Backward pass
loss.backward()

# print out the gradient
print("dL/dw : ", linear.weight.grad)
print("dL/db : ", linear.bias.grad)

# 1-step gradient descent.
optimizer.step()

# Print out the loss after 1-step gradient descent.
pred = linear(x)
loss = criterion(pred, y)
print('loss after 1 step optimization: ', loss.item())



#################################################################
#             Loading data from numpy and numpy to tensor
#################################################################

# Create a numpy array.
x = np.array([[1, 2], [3, 4]])
print("numpy array is :", x)

# Convert the numpy array to a torch tensor.
y = torch.from_numpy(x)
print("tensors are :", y)

# Convert the torch tensor to a numpy array.
z = y.numpy()
print("Numpy array is :", z)



# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Download and load the pretrained ResNet-18.
resnet = torchvision.models.resnet18(pretrained=True)

# If you want to finetune only the top layer of the model, set as below.
for param in resnet.parameters():
    param.requires_grad = False

# Replace the top layer for finetuning.
resnet.fc = nn.Linear(resnet.fc.in_features, 100)  # 100 is an example.

# Forward pass.
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print("Output size is :", outputs.size())     # (64, 100)



# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
