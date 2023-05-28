from torch import nn

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.C1 = nn.Conv2d(1,6,5)
        self.Tanh1 = nn.Tanh()
        self.S2 = nn.MaxPool2d(2)
        self.C3 = nn.Conv2d(6,16,5)
        self.Tanh2 = nn.Tanh()
        self.S4 = nn.MaxPool2d(2)
        self.FC5 = nn.Linear(16*5*5,120)
        self.Tanh3 = nn.Tanh()
        self.FC6 = nn.Linear(120,84)
        self.Tanh4 = nn.Tanh()
        self.FC7 = nn.Linear(84,50)

    def forward(self,X):

        X = self.C1(X)
        X = self.Tanh1(X)
        X = self.S2(X)
        X = self.C3(X)
        X = self.Tanh2(X)
        X = self.S4(X)
        X = X.view(X.shape[0], -1)
        X = self.FC5(X)
        X = self.Tanh3(X)
        X = self.FC6(X)
        X = self.Tanh4(X)
        X = self.FC7(X)
        return X