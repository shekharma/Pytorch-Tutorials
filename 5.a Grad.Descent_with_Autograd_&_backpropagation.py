## Link for video https://youtu.be/E-I2DNVzQLg?feature=shared
import numpy as np

#f=w*x
#f=2*x
x=np.array([1,2,3,4],dtype=np.float32)
y=np.array([2,4,6,8],dtype=np.float32)

w=0.0

#model prediction
def forward(x):
    return w*x

#loss=MSE

def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

#gradient
#MSE=1/N*(w*x-y)**2
## dj/dw=1/N 2x (w*x-y)

def gradient(x,y,y_pred):
    return np.dot(2*x,y_pred-y).mean()

print(f'Prediction before training:f(5) = {forward(5):.3f}')

##training
lr=0.01
n_iters=10

for epoch in range(n_iters):
    #prediction =forward pass
    y_pred=forward(x)

    #loss
    l=loss(y,y_pred)

    #gradients
    dw=gradient(x,y,y_pred)

    #update weights
    w-=lr*dw

    if epoch%1==0:
        print(f'epoch{epoch+1}:w={w:.3f}, loss={l:.4f}')
print(f'Prediction before training:f(5) = {forward(5):.3f}')

## From result we can see that the our initial guess w=0.0 and loss is 30
## but as we start training loss goes down and w start to approach its original value