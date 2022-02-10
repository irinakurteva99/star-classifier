import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from config import sampleSize
from dataset import StarDataset

class Net(nn.Module):
    hiddenSize = 3000

    def __init__(self):
        super(Net, self).__init__()
        self.inputLayer = nn.Linear(sampleSize, self.hiddenSize).double()
        self.outputLayer = nn.Linear(self.hiddenSize, 1).double()

    def forward(self, x):
        x = torch.sigmoid(self.inputLayer(x))
        x = self.outputLayer(x)
        x = torch.flatten(x)
        return x

class MyNetwork:

    def __init__(self):
        self.net = Net()

        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.scheduler = ReduceLROnPlateau(self.optimizer, verbose=True)

        self.transform = lambda star: (star.data, float(star.temp) / 10000)
        self.trainSet = StarDataset("/home/ikurteva/ai/stars/train", self.transform)

        self.trainLoader = DataLoader(self.trainSet, batch_size=10, shuffle=False)

    def train(self):
        for epoch in range(100):
            avgLoss = 0
            avgC = 0
            for flux, temp in self.trainLoader:
                self.optimizer.zero_grad()
                output = self.net(flux)

                loss = self.criterion(output, temp)
                avgLoss += float(loss)
                avgC += 1
                # print(output, temp)
                # print(loss)

                loss.backward()
                self.optimizer.step()
    
            avgLoss /= avgC
            print(avgLoss)
            self.scheduler.step(loss)

if __name__ == '__main__':
    MyNetwork().train()
