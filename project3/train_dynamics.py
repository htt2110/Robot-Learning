import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torch.optim as optim
import argparse
import time
from torch.autograd import Variable

np.set_printoptions(suppress=True)
random.seed(0)
class DynamicDataset(Dataset):
    def __init__(self, dataset_dir):
        # X: (N, 9), Y: (N, 6)
        self.X = np.load(os.path.join(dataset_dir, 'X.npy')).T.astype(np.float32)
        self.Y = np.load(os.path.join(dataset_dir, 'Y.npy')).T.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class Net(nn.Module):
    # ---
    # Your code goes here
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim,32)
        self.fc2 = nn.Linear(32,64)
        self.fc3 = nn.Linear(64, 256)
        self.fc4 = nn.Linear(256,64)
        self.fc5 = nn.Linear(64,32)
        self.fc6 = nn.Linear(32,6)
        
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

    # ---

def train(model):
    model.train()

    # ---
    model = Net(9)
    trainer = main.model()
    features = train_set[:][0]
    labels = train_set[:][1]
    trainer.train(labels, features)
    # ---


def test(model):
    model.eval()

    # --
    # Your code goes here
    
    # ---

    return test_loss


def get_args():
    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True)
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--split', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr)
    return args


def main():
    args = get_args()
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = DynamicDataset(args.dataset_dir)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 150)
    test_loader = torch.utils.data.DataLoader(test_set)
    
    
    # ---
    # Your code goes here  
    model = Net(9)
    
    num_epochs = 300
    lr = 0.0001
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
            
    def train_epoch(train_loader):
        total_loss = 0.0
        for i, train_set in enumerate(train_loader, 0):
            features = train_set[:][0].float()
            labels = train_set[:][1].float()
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        print ('loss', total_loss/i)    
        model_folder_name = f'epoch_{epoch:04d}_loss_{total_loss:.8f}'
        if not os.path.exists(os.path.join(args.save_dir, model_folder_name)):
            os.makedirs(os.path.join(args.save_dir, model_folder_name))
        torch.save(model.state_dict(), os.path.join(args.save_dir, model_folder_name, 'dynamics.pth'))
        print(f'model saved to {os.path.join(args.save_dir, model_folder_name, "dynamics.pth")}\n')


    for epoch in range(num_epochs):
        train_epoch(train_loader)
                    
    for i, test_set in enumerate(test_loader, 0):
            test_features = test_set[:][0].float()
            preds = model(test_features).float()
    test_loss = criterion(preds, test_set[:][1])
    print(test_loss)
    # ---


if __name__ == '__main__':
    main()
