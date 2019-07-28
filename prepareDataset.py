import cv2
import os as os
from os import listdir
from os.path import isfile, join
import json
import pickle
import torch
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import torch.nn as nn
import random
import time
from dataset import *
DATASET_PATH = "data/"
SIZE = 64

dataset = EmojiDataset()
files = [f for f in listdir(DATASET_PATH)]
for name in files:
    if name !=  '.DS_Store':
        openName = DATASET_PATH + name
        img = cv2.imread(openName)
        height, width = img.shape[:2]
        scalingFactor = SIZE / float(height)
        img = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
        img = (img / 255 - 0.5) * 2
        sample = torch.from_numpy(img).type(torch.float)
        dataset.add(sample)


dataLoader = MyLoader(dataset, 0, len(dataset) - 1)
saveToFile(dataLoader, "dataLoader.pkl")
