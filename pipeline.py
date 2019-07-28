from config import *
from dataset import *
from model import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2



"""
    in perspective to make values of this labels undeterministic
    i.e. add noise which could help with convergence
"""


realLabel = 1
fakeLabel = 0

def noisyRealLabel():
    return np.random.uniform(0.7, 1.2)

def noisyFakeLabel():
    return np.random.uniform(0.0, 0.3)


def noisyRealLabels(size):
    return torch.tensor([noisyRealLabel() for i in range(size)])

def noisyFakeLabels(size):
    return torch.tensor([noisyFakeLabel() for i in range(size)])


generatorNet = Generator(64).to(device)
dicriminatorNet = Discriminator(16).to(device)


generatorNet.apply(weights_init)
dicriminatorNet.apply(weights_init)


criterion = nn.BCELoss()

dicriminatorOptimizer = optim.Adam(dicriminatorNet.parameters(),
                                   lr = learningRate, betas = (beta1, 0.999))
generatorOptimizer = optim.Adam(generatorNet.parameters(),
                                lr = learningRate, betas = (beta1, 0.999))

dataLoader = loadFromFile("dataLoader.pkl")


"""
    Info track section
"""
generatorErrors = []
dicriminatorErrors = []
iter = 0
fixedNoise = torch.randn(1, 100, 1, 1, device = device)
generatedImagePath = "genImg/"
"""
"""
generatorNet.train()
dicriminatorNet.train()
for e in range(config.numEpochs):
    print("---epoch-[", e,  "]---begin")
    for i in range(config.numIterations):
        """
            First step is to perform all-reall batch pass through
            Discriminator network and hold gradients to update them later
        """
        dicriminatorNet.zero_grad()
        realDataBatch = dataLoader.getBatch().to(device, dtype = dtype)
        labels = noisyRealLabels(config.batchSize)
        predictions = dicriminatorNet(realDataBatch).view(-1)
        errorDReal = criterion(predictions, labels)
        meanDResult = predictions.mean().item()
        """
            We perform only gradient flow yet, without weights update
        """

        """
            Second step is to perform all-fake batch pass through Discriminator net
            all-fake batch after Generator net
            and only after all-fake pass we perform gradient update
            but only for Discriminator
            
            --  usage of detach could be a bit confusing, but this is done
                with the aim of not updating weights of generator here
        """
        noiseDNA = torch.randn(config.batchSize, 100, 1, 1, device = device)
        fakeImage = generatorNet(noiseDNA)
        
        #print(fakeImage.mean().item(), fakeImage.min().item(), fakeImage.max().item())
        
        labels = noisyFakeLabels(config.batchSize)
        anotherPredictions = dicriminatorNet(fakeImage.detach()).view(-1)
        errorDFake = criterion(anotherPredictions, labels)
        totalDError = errorDReal + errorDFake
        totalDError.backward()
        dicriminatorOptimizer.step()
        D_G_z1 = anotherPredictions.mean().item()
        """
            Third step is to update Generator network
            
            -- The purpose of Generator is to fool Discriminator
        """
        generatorNet.zero_grad()
        labels = noisyRealLabels(config.batchSize)  # Generator "thinks" that it is creates real images
        againPredictions = dicriminatorNet(fakeImage).view(-1)
        errorGen = criterion(againPredictions, labels)
        errorGen.backward()
        generatorOptimizer.step()
        D_G_z2 = againPredictions.mean().item()
        """
            Info tracking Staff
        """
        iter += 1
        generatorErrors.append(errorGen.item())
        dicriminatorErrors.append(totalDError.item())

        # Just statistics
        if (iter % 50 == 0):
            print("epoch - [%d]  iter - [%d]  D(x) - [%.4f]   LossD - [%.4f]   LossG - [%.4f]   D(G(z)) : %.4f / %.4f" % (e, iter, meanDResult, totalDError.mean().item(), errorGen.item(), D_G_z1, D_G_z2))
        


        if (iter % 100 == 0):
            with torch.no_grad():
                fakeImage = generatorNet(fixedNoise)
                fakeImage = ((fakeImage / 2) + 0.5) * 255
                img = fakeImage.cpu().view(32,32,3).detach().numpy().astype(np.uint8)
                cv2.imwrite(generatedImagePath + "img" + str(iter) + ".png", img)
                torch.save(generatorNet.state_dict(), "Generator")
                torch.save(dicriminatorNet.state_dict(), "Discriminator")
