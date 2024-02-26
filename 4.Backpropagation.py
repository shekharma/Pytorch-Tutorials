# Link for video  - https://youtu.be/3Kb0QS6z7WA?feature=shared

## Backpropagation use chain rule  (dz/dx=dz/dy*dy/dx)  on computation graph i.e calculate local gradient
## Here is main idea is to calculate the derivative of loss w.r.t. every weight/paramater intuition is to 
## check the impact of particular weight on loss
### Squared loss = (y_pred-y)^2
## steps
## 1st forward pass - calcuklate y_pred
## 2nd calculate loss
## 3rd Backpropagation - calculate d(loss)/d(parameter)


import torch
x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)
## forward pass and compute the loss
y_hat=w*x
loss=(y_hat-y)**2
print(loss)

# backward pass
loss.backward()
print(w.grad)

## based on gradient we update the wieight and repeats the process