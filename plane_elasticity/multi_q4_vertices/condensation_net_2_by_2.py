import torch
import torch.nn as nn
import torch.nn.functional as F

class CondensationNet2by2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 50, bias=True) # Fully connected layer. Input size is 4 (4 Young modulus of the 2x2 grid)
        self.fc2 = nn.Linear(50, 50, bias=True) # Fully connected layer 
        self.fc3 = nn.Linear(50, 50, bias=True) # Fully connected layer
        self.fc4 = nn.Linear(50, 50, bias=True) # Fully connected layer
        self.fc5 = nn.Linear(50, 50, bias=True) # Fully connected layer
        self.fc6 = nn.Linear(50, 50, bias=True) # Fully connected layer
        self.fc7 = nn.Linear(50, 80, bias=True) # Fully connected layer. Output size is 4 (Matrix K_bb\Kba has size 10x8)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.elu(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = F.elu(self.fc6(x))
        x = self.fc7(x)
        return x