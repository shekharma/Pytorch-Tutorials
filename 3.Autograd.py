## Link for video   https://youtu.be/DbeIqrwb_dE?feature=shared

import torch

x=torch.randn(3, requires_grad=True)
print(x)
y=x+2
# 1st forward pass
# 2nd calculate y_pred, loss
# 3rd backpropagate

print(y)
z=y*y*2
#z=z.mean()
print(z)
#z.backward()#calculate dz/dx
#print(x.grad)
## In backgroud we have a jacobian amtrix with partial derivative(dy_i/dx_i)
## multiply with gardient vector and this called chain rule.

##Note:- Grad is only created for scalar value that's why we used z.mean()
## and is it is not a scalar then we have mention is explicitly

v=torch.tensor([0.1,1,0.001],dtype=torch.float32)
z.backward(v)
print(x.grad)

##########################################################
## To remove require_grad attribute
# 1st x.requires_grad_(False)
# 2nd x.detach()
# 3rd with torch.no_grad():
#         y=x+2
#         print(y)
########################################################3
import torch
weights=torch.randn(4, requires_grad=True)
for epoch in range(3):
    model_output=(weights*3).sum()
    model_output.backward()
    print(weights.grad)
    weights.grad.zero_() ## this avoid the cummulation of gradient


##############################################
##########   OPTIMIZATION  ##################
import torch
weights=torch.ones(4, requires_grad=True)
optimizer=torch.optim.SGD(weights, lr=0.01)
optimizer.step()
optimizer.zero_grad()

###############################################