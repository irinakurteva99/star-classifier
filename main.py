from models import Star, ProtoStar
from dataset import StarDataset

def main():
    dataset = StarDataset("/home/ikurteva/ai/testDir")
    for i in dataset:
        print(i)
