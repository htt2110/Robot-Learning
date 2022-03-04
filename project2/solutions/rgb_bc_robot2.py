from base import RobotPolicy
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

torch.manual_seed(0)
class RGBBCRobot2(RobotPolicy):

    """ Implement solution for Part3 below """

    def train(self, data):
        #for key, val in data.items():
        #    print(key, val.shape)
        #print("Using dummy solution for RGBBCRobot2")
        #pass
        self.network = CNN()
        trainer = MyDNNTrain(self.network)
        features = data['obs'].swapaxes(1,3).swapaxes(2,3)
        labels = np.ravel(data['actions'])
        trainer.train(labels, features)

    def get_action(self, obs):
        obs = obs.reshape(1,64,64,3).swapaxes(1,3).swapaxes(2,3)
        out = self.network.predict(obs)
        preds = np.where(out == np.max(out))
        #print(preds[1][0])
        return preds[1][0]
    	
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 15, 4)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(15,60, 4)
        self.fc1 = nn.Linear(60*13*13, 2048)
        self.fc2 = nn.Linear(2048,1024)
        self.fc3 = nn.Linear(1024,512)
        self.fc4 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 60 * 13* 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
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
        self.learning_rate = 1
        self.optimizer = torch.optim.Adadelta(self.network.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.num_epochs = 15
        self.batchsize = 90
        self.shuffle = True

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
