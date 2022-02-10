import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import sys

from config import sampleSize
from dataset import StarDataset

class Net(nn.Module):
    hiddenSize = 3000

    def __init__(self):
        super(Net, self).__init__()
        self.inputLayer = nn.Linear(sampleSize, self.hiddenSize).double()
        self.middleLayer = nn.Linear(self.hiddenSize, self.hiddenSize).double()
        self.middleLayer2 = nn.Linear(self.hiddenSize, self.hiddenSize).double()
        self.outputLayer = nn.Linear(self.hiddenSize, 1).double()

    def forward(self, x):
        x = torch.sigmoid(self.inputLayer(x))
        x = torch.sigmoid(self.middleLayer(x))
        x = torch.sigmoid(self.middleLayer2(x))
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
        self.testSet = StarDataset("/home/ikurteva/ai/stars/test", self.transform)

        self.trainLoader = DataLoader(self.trainSet, batch_size=32, shuffle=True)
        self.testLoader = DataLoader(self.testSet, batch_size=32, shuffle=True)

    def train(self):
        for epoch in range(1000):
            for flux, temp in self.trainLoader:
                self.optimizer.zero_grad()
                output = self.net(flux)

                loss = self.criterion(output, temp)
                # print(output, temp)
                # print(loss)

                loss.backward()
                self.optimizer.step()
    
            self.scheduler.step(loss)
            if epoch % 5 == 0:
                self.test(self.trainLoader, "train")
                self.test(self.testLoader, "test")

    def test(self, data, typ):
        lossfun = nn.MSELoss()
        avgLoss = 0
        avgC = 0
        accuracyEls = 0
        accuracy = 0
        bigAcc = 0
        for flux, temp in data:
            output = self.net(flux)
            # if avgC == 0:
            #     print(output, temp)
            loss = lossfun(output, temp)
            avgLoss += float(loss)
            avgC += 1
            dists = torch.abs(output - temp)
            accuracy += len(dists[dists < 500 / 10000])
            bigAcc += len(dists[dists < 1000 / 10000])
            accuracyEls += len(output)
        avgLoss /= avgC
        print('{} Accuracy +-500: {:.2f}% Accuracy +-1000: {:.2f}%'.format(typ, accuracy / accuracyEls * 100, bigAcc / accuracyEls * 100))
        print('Avg {} Loss: {}'.format(typ, avgLoss))
        sys.stdout.flush()

if __name__ == '__main__':
    MyNetwork().train()
