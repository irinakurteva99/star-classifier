import torch
import torch.nn as nn
import torch.optim as optim
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

def do_stuff2():
    net = Net()
    print(net)

    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    transform = lambda star: (star.data, float(star.temp) / 10000)
    train = StarDataset("/home/ikurteva/ai/stars/train", transform)

    trainLoader = DataLoader(train, batch_size=10, shuffle=False)
    
    for epoch in range(100):
        avgLoss = 0
        avgC = 0
        for flux, temp in trainLoader:
            optimizer.zero_grad()
            output = net(flux)
   
            loss = criterion(output, temp)
            avgLoss += float(loss)
            avgC += 1
            # print(output, temp)
            # print(loss)

            loss.backward()
            optimizer.step()

        avgLoss /= avgC
        print(avgLoss)

if __name__ == '__main__':
    do_stuff2()
