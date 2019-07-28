import torch
import random
from random import shuffle
import config
import pickle


def saveToFile(object, filename):
    with open(filename, "wb") as file:
        pickle.dump(object, file)

def loadFromFile(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)



"""
    Data Holder class
"""
class EmojiDataset:
    def __init__(self):
        self.data = []

    def getOne(self, index):
        return self.data[index]

    def add(self, picture):
        self.data.append(picture)

    def __len__(self):
        return len(self.data)

    def shuffle(self):
        indicies = [i for i in range(0, len(self.data))]
        shuffle(indicies)
        
        dataNew = []
        
        for i in indicies:
            dataNew.append(self.data[i])
        
        self.data = dataNew



class DataLoader:
    def __init__(self, dataset, indexMin, indexMax):
        assert(indexMax > indexMin)
        self.dataset = dataset
        self.allIndicies = [i for i in range(indexMin, indexMax + 1)]
    
    
    def getBatch(self):
        indcies = random.sample(self.allIndicies, config.batchSize)
        data = [self.dataset.getOne(i) for i in indicies]
        return torch.tensor(data)
    
    def __len__(self):
        return len(self.dataset)

