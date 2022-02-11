from matplotlib import pyplot as plt
import pickle
import numpy


class Star:
    def __init__(self, temp, data):
        self.temp = temp
        self.data = data

    def __repr__(self):
        return "Star(Temp: {} Data: {}...)".format(
                self.temp, ', '.join([str(x) for x in self.data[0:5]])
        )

    def plot(self):
        plt.title("Star plot")
        plt.xlabel("Index")
        plt.ylabel("Flux")
        plt.plot(numpy.arange(len(self.data)),self.data)
        plt.savefig('star.png')
        plt.clf()

        with open('/tmp/norm_data', "rb") as f:
            (meanT, stdT, meanF, stdF) = pickle.load(f)

        plt.title("Star plot (normalised)")
        plt.xlabel("Index")
        plt.ylabel("Normalised flux")
        plt.plot(numpy.arange(len(self.data)), (self.data - meanF) / stdF)
        plt.savefig('star_normalised.png')
        plt.clf()

    def save(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self,f)

    @classmethod
    def load(cls, filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

class ProtoStar:
    def __init__(self, temp, data):
        self.temp = temp
        self.spec = data[:,1]
        self.start = data[0,0]
        self.wavelen = data[:,0]
        self.length = len(self.wavelen)

    def __repr__(self):
        return "ProtoStar(Temp: {} Start: {} Len: {})".format(self.temp, self.start, self.length)

    def plot(self):
        plt.title("Protostar plot")
        plt.xlabel("Angstooioms")
        plt.ylabel("Flux")
        plt.plot(self.wavelen,self.spec)
        plt.savefig('protostar.png')
        plt.clf()

