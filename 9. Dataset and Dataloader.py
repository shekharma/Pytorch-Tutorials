# IMPORTANT POINTS
# 1) Epoch = one forward and one backward pass
# 2) batch_size= no. of training samples in one forward and backward pass
# 3) iteration = no.of passes, each pass using [batch_size] number of samples
# eg. samples=100 batch_size=20 then for 1 epoch 100/20 there will be 5 iteration 

import torch
import torchvision  ## we can get some inbuilt dataset such as fashion-mnist,cifar, coco
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class winedataset(Dataset):
    def __init__(self):
        # data  loading
        xy=np.loadtxt('./data/wine/wine.csv', delimiter=',',dtype=np.float32,skiprows=1)
        self.x=torch.from_numpy(xy[:,1:])
        self.y=torch.from_numpy(xy[:,[0]]) ## n_samples,1
        self.n_samples=xy.shape[0]

    def __getitem__(self, index):
        return self.x[index],self.y[index]
    
    def __len__(self):
        return self.n_samples
    
dataset=winedataset()
#### Dataset view  ###
#first_data=dataset[0]
#features, labels= first_data  ## this unpack our data and gives us a features and label
#print(features,labels)

## below 4 line will print your 4 batches and their corresponding labels
dataloader= DataLoader(dataset=dataset, batch_size=4, shuffle=True,num_workers=2)
dataiter=iter(dataloader)
data=dataiter.next()
features, labels= data
print(features,labels)

# this training loop iterate over a data and print epoch and iterations
# training loop
num_epochs=2
total_samples=len(dataset)
n_iterations=math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs,labels) in enumerate(dataloader):
        #forward backward, update
        if (i+1)%5==0:
            print(f'epoch{epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')