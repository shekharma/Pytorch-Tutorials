#Link for video https://youtu.be/YAJ5XBwlN4o?feature=shared
# There are three stages in Model pipeline
#1. Design model (input, ouput size , forward pass)
#2. Construct loss and optimizer
#3. Training loop

# forward pass =compute prediction
# backward pass - gradients
# update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


# 0) prepare data- the below library prepare data by randomly sampling 
x_numpy, y_numpy=datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=1)

x=torch.from_numpy(x_numpy.astype(np.float32))
y=torch.from_numpy(y_numpy.astype(np.float32))
y=y.view(y.shape[0],1)
n_samples,n_features=x.shape

# 1) Model
input_size=n_features
output_size=1
model=nn.Linear(input_size,output_size)
# 2) loss and Optimizer
lear_rate=0.01
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=lear_rate)

# 3) training loop
num_epochs=10
for epoch in range(num_epochs):
    # forward pass
    y_pred=model(x)
    loss=criterion(y_pred,y)

    #backward pass
    loss.backward()

    #update
    optimizer.step()
    optimizer.zero_grad()
    if (epoch+1)%1==0:
        print(f'epoch:{epoch+1},loss={loss.item():.4f}')
#plot 
predicted=model(x).detach().numpy() ##this detach command will generate new tensor where our require_grad=False
plt.plot(x_numpy,y_numpy,'ro')
plt.plot(x_numpy,predicted,'b')
plt.show()
