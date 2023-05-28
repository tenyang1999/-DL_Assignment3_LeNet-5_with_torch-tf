import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1,6,5)
        self.ReLU1 = nn.ReLU()
        self.S2 = nn.MaxPool2d(2)
        self.C3 = nn.Conv2d(6,16,5)
        self.ReLU2 = nn.ReLU()
        self.S4 = nn.MaxPool2d(2)
        self.FC5 = nn.Linear(16*5*5,120)
        self.ReLU3 = nn.ReLU()
        self.FC6 = nn.Linear(120,84)
        self.ReLU4 = nn.ReLU()
        self.FC7 = nn.Linear(84,50)
        
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self,X):
        X = self.quant(X)
        X = self.C1(X)
        X = self.ReLU1(X)
        X = self.S2(X)
        X = self.C3(X)
        X = self.ReLU2(X)
        X = self.S4(X)
        X = X.view(X.shape[0], -1)
        X = self.FC5(X)
        X = self.ReLU3(X)
        X = self.FC6(X)
        X = self.ReLU4(X)
        X = self.FC7(X)
        X = self.dequant(X)
        return X
    
    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['C1', 'Sigmoid1'],['C3', 'Sigmoid2'],['FC5', 'ReLU3'],['FC6', 'ReLU4']], inplace=True) #