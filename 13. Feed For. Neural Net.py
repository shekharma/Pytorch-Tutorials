#MNISt
#Dataloader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training loop (Batch Training)
# Model evaluation
# GPU Support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device config
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
input_size=784 # 28x28
hidden_size=100
num_classes=10
num_epoch=2
batch_size=100
learning_rate=0.001

#MNIST
train_dataset=torchvision.datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset=torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

train_loader=torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

examples=iter(train_loader)
#for samples, labels in examples:
#    print(samples.shape, labels.shape)
#    pass
#print(examples.shape)
samples, labels=next(examples)
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(samples[i][0], cmap='gray')

plt.show()

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet,self).__init__()
        self.l1=nn.Linear(input_size,hidden_size)
        self.relu=nn.ReLU()
        self.l2=nn.Linear(hidden_size,num_classes)

    def forward(self,x):
        out=self.l1(x)
        out=self.relu(out)
        out=self.l2(out)

        return out
model=NeuralNet(input_size,hidden_size, num_classes).to(device)

#loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps=len(train_loader)
for epoch in range(num_epoch):
    for i, (images, labels) in enumerate(train_loader): # enumerate give you the index
        # 100, 1, 28, 28
        # 100(batches), 784(image_shape)
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        #forward pass
        outputs =model(images)
        loss=criterion(outputs,labels)
        
        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100==0:
            print(f'epoch {epoch+1}/{num_epoch},step {i+1}/{n_total_steps},loss={loss.item():.4f}')

# test and evaluation
with torch.no_grad():
    n_correct=0
    n_samples=0
    for images, labels in test_loader:
        images=images.reshape(-1,28*28).to(device)
        labels=labels.to(device)

        outputs=model(images)
        #value, index
        _, predictions=torch.max(outputs,1)
        n_samples+=labels.shape[0]

        n_correct=(predictions==labels).sum().item() # for each correct prediction will add 1
    acc=100*n_correct/n_samples
    print(f'accuracy={acc}')




