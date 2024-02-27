#Link for video https://youtu.be/OGpQxIkR4ao?feature=shared
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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data
breast_cancer=datasets.load_breast_cancer()
x,y=breast_cancer.data, breast_cancer.target
n_samples, n_features=x.shape
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2, random_state=1)

## scale
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


x_train=torch.from_numpy(x_train.astype(np.float32))
x_test=torch.from_numpy(x_test.astype(np.float32))
y_train=torch.from_numpy(y_train.astype(np.float32))
y_test=torch.from_numpy(y_test.astype(np.float32))


y_train=y_train.view(y_train.shape[0],1)
y_test=y_test.view(y_test.shape[0],1)

# 1) model
# f=wx+b,sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression,self).__init__()
        self.linear=nn.Linear(n_input_features,1)## this command will take n input features and predict one class

    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
    
model=LogisticRegression(n_features)

# 2) loss and optimizer
learning_rate=0.01
criterion =nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)

# training loop 
num_epochs=10
for epoch in range(num_epochs):
    #forward pass
    y_pred=model(x_train)

    loss=criterion(y_pred,y_train)

    #backward pass
    loss.backward()
    optimizer.step()

    ## zero gradients
    optimizer.zero_grad()

    if (epoch+1)%1==0:
        print(f'epoch:{epoch+1}, loss={loss.item():.4f}')

with torch.no_grad():
    y_pred=model(x_test)
    y_pred_cls=y_pred.round()
    acc=y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy={acc:.4f}')
