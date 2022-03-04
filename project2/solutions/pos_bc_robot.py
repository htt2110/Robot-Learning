from base import RobotPolicy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

import math
torch.manual_seed(42)
class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part1 below """
    
    
    def train(self, data):
        #for key, val in data.items():
        #    print(key, val.shape)
        #print("Using dummy solution for POSBCRobot")
        #pass
        self.network = DNN(4)
        trainer = MyDNNTrain(self.network)
        features = data['obs']
        labels = np.ravel(data['actions'])
        trainer.train(labels, features)



    def get_action(self, obs):
        out = self.network.predict(obs)
        preds = np.where(out == np.max(out))
        #print(preds)
        return preds[0][0]

class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128,64)
        self.fc6 = nn.Linear(64, 4)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x
        
    def predict(self, features):
        self.eval()	#Sets network in eval mode (vs training mode)
        features = torch.from_numpy(features).float()
        return self.forward(features).detach().numpy()

class MyDataset(Dataset):
    def __init__(self, labels, features):
        super(MyDataset, self).__init__()
        self.labels = labels
        self.features = features

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):		#This tells torch how to extract a single datapoint from a dataset, Torch randomized and needs a way to get the nth-datapoint
        feature = self.features[idx]
        label = self.labels[idx]
        return {'feature': feature, 'label': label}
        
class MyDNNTrain(object):
    def __init__(self, network):	#Networks is of datatype MyDNN
        self.network = network
        self.learning_rate = 0.1
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.learning_rate, momentum = 0.9)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 300 #300 Epochs for full points
        self.batchsize = 150
        self.shuffle = False

    def train(self, labels, features):
        self.network.train()
        dataset = MyDataset(labels, features)
        loader = DataLoader(dataset, shuffle=self.shuffle, batch_size = self.batchsize)
        for epoch in range(self.num_epochs):
            self.train_epoch(loader)

    def train_epoch(self, loader):
        total_loss = 0.0
        for i, data in enumerate(loader, 0):
            features = data['feature'].float()
            labels = data['label']
            self.optimizer.zero_grad()
            predictions = self.network(features)
            loss = self.criterion(predictions, labels)
            loss.backward()
            total_loss += loss.item()
            self.optimizer.step()
        print ('loss', total_loss/i)



