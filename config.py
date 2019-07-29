import os as os
from os import listdir
from os.path import isfile, join
import json
import torch


device = 0
dtype = torch.float32

batchSize = 32
batchShape = (batchSize, 3, 32, 32)
imageShape = (32,32,3)

numEpochs = 20
numIterations = 10000
beta1 = 0.5

learningRate = 0.0002
