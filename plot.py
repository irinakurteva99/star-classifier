import sys

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import StarDataset

def plotTemps(starDir):
    transform = lambda star: (star.data, float(star.temp))
    trainSet = StarDataset(starDir, transform)

    trainLoader = DataLoader(trainSet, batch_size=32, shuffle=False)

    temps = []
    for _,temp in trainLoader:
       temps += temp.tolist()
    
    n, bins, patches = plt.hist(x=temps, bins=100, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig('M.png')

if __name__ == '__main__':
    plotTemps(sys.argv[1])
