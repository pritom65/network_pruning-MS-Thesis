import os
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt

class Mito_Dataset(Dataset):
    def __init__(self,path,transform=None):
        self.path = path
        self.data_path = os.listdir(self.path)
        self.data_path.sort()
        self.transform = transform
    
    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, index):
        X = np.load(self.path + self.data_path[index])
        X = torch.Tensor(X)
        if self.transform:
            X = self.transform(X)
            X,y = X[[0]],X[[1]]
            y = torch.round(y)
        
        return X,y

class Standarization(nn.Module):
    def forward(self,x):
        return x/255
    
augmentation = T.Compose([
    Standarization(),
    T.RandomResizedCrop(size=(256,256),scale=(.4,1)),
    T.RandomVerticalFlip(p=.5),
    T.RandomHorizontalFlip(p=.5),
])

val_augmentation = T.Compose([
    Standarization(),
    T.Resize((256,256))
])


def show_image(X,y):
    fig,ax = plt.subplots(1,2)
    ax[0].imshow(X.to('cpu')[0])
    ax[1].imshow(y.to('cpu')[0])
    for i in ax:
        i.axis('off')
    plt.tight_layout()
    plt.show()