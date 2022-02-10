from models import Star, ProtoStar
from dataset import StarDataset
import config

import torch
import torch.nn as nn
import numpy as np

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim).double()
        self.sigmoid1 = nn.Sigmoid().double()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim).double()
        self.sigmoid2 = nn.Sigmoid().double()
        self.fc3 = nn.Linear(hidden_dim, hidden_dim).double()
        self.sigmoid3 = nn.Sigmoid().double()
        self.fc4 = nn.Linear(hidden_dim, output_dim).double()
        self.flattenisinator = nn.Flatten(start_dim=0).double()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid1(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        out = self.fc3(out)
        out = self.sigmoid3(out)
        out = self.fc4(out)
        out = self.flattenisinator(out)
        return out

def do_stuff():
    transform = lambda star: (star.data, star.temp/10000)
    train_dataset = StarDataset("/home/ikurteva/ai/stars/train", transform)
    test_dataset = StarDataset("/home/ikurteva/ai/stars/test", transform)

    batch_size = 10 # increase after dataset size increases
    num_epochs = 400 
    n_iters = int(num_epochs * (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    input_dim = config.sampleSize
    hidden_dim = 3000
    output_dim = 1

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)
    model.zero_grad()
    criterion = nn.MSELoss()

    learning_rate = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter = 0
    for epoch in range(num_epochs):
        for i, (flux, temp) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(flux.double())
            
            loss = criterion(outputs, temp.double())

            loss.backward()

            optimizer.step()

            iter += 1

            if iter % 100 == 0:
                print('Last loss: {}'.format(loss))
                print('Train')
                testErr(train_loader, model)
                print('Test')
                testErr(test_loader, model)

def testErr(loader, model):
    errFun = nn.MSELoss()
    maxErr = 0
    errs = []
    for flux, temp in loader:
        outputs = model(flux.double())
        err = errFun(outputs, temp.double())
        errs.append(err)
 
    # Print Loss
    print('Avg Err: {} MaxErr: {}'.format(sum(errs)/len(errs), max(errs)))

def main():
    dataset = StarDataset("/home/ikurteva/ai/testDir")
    for i in dataset:
        print(i)

if __name__ == '__main__':
    do_stuff()
