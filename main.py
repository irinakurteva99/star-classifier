from models import Star, ProtoStar
from dataset import StarDataset
import config

import torch
import torch.nn as nn
import numpy as np

class FeedforwardNeuralNetModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedforwardNeuralNetModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.flattenisinator = nn.Flatten(start_dim=0)

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.flattenisinator(out)
        return out

def do_stuff():
    transform = lambda star: (star.data, star.temp)
    train_dataset = StarDataset("/home/ikurteva/ai/stars/train", transform)
    test_dataset = StarDataset("/home/ikurteva/ai/stars/test", transform)

    batch_size = 10 # increase after dataset size increases
    num_epochs = 500
    n_iters = int(num_epochs * (len(train_dataset) / batch_size))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    input_dim = config.sampleSize
    hidden_dim = 100
    output_dim = 1

    model = FeedforwardNeuralNetModel(input_dim, hidden_dim, output_dim)

    criterion = nn.MSELoss()

    learning_rate = 0.1

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    iter = 0
    for epoch in range(num_epochs):
        for i, (flux, temp) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(flux.float())

            loss = criterion(outputs, temp.float())
            loss.backward()

            optimizer.step()

            iter += 1

            if iter % 500 == 0:
                maxErr = 0
                for flux, temp in test_loader:
                    outputs = model(flux.float())

                    err = torch.max(torch.abs(outputs.data - temp.float()))
                    maxErr = max(maxErr, err)

                # Print Loss
                print('Iteration: {}. MaxErr: {}'.format(iter, maxErr))

def main():
    dataset = StarDataset("/home/ikurteva/ai/testDir")
    for i in dataset:
        print(i)

do_stuff()
