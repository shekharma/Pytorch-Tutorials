#Link for video https://youtu.be/VVDHU_TWwUg?feature=shared
# There are three stages in Model pipeline
#1. Design model (input, ouput size , forward pass)
#2. Construct loss and optimizer
#3. Training loop

# forward pass =compute prediction
# backward pass - gradients
# update weights
import torch
import torch.nn as nn

x=torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y=torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

x_test=torch.tensor([5],dtype=torch.float32)
n_samples, n_features=x.shape
print(n_samples,n_features)
input_size=n_features
output_size=n_features
#model=nn.Linear(input_size, output_size)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression,self).__init__()
        #define layers
        self.lin=nn.Linear(input_dim,output_dim)

    def  forward(self,x):
        return self.lin(x)
model=LinearRegression(input_size,output_size)

print(f'Prediction before training:f(5) = {model(x_test).item():.3f}')

##training
lr=0.01
n_iters=20

loss=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=lr)


for epoch in range(n_iters):
    #prediction =forward pass
    y_pred=model(x)

    #loss
    l=loss(y,y_pred)

    #gradients=backward pass
    #dw=gradient(x,y,y_pred)
    l.backward()#dl/dw

    #update weights
    optimizer.step()
    #zero gradients
    optimizer.zero_grad()
    if epoch%1==0:
        [w,b]=model.parameters()
        print(f'epoch{epoch+1}:w={w[0][0].item():.3f}, loss={l:.4f}')
print(f'Prediction before training:f(5) = {model(x_test).item():.3f}')
