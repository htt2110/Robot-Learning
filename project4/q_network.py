import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, env):
        super(QNetwork, self).__init__()
        # ---
        # Define your Q network here
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,32)
        self.fc3 = nn.Linear(32,64)
        self.fc4 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,16)
        self.fc6 = nn.Linear(16,3)
        # ---

    def forward(self, x, device):
        # ---
        # Write your forward function to output a value for each action
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))        
        x = self.fc6(x)
        # ---
        return x
