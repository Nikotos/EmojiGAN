import torch
import torch.nn as nn
import config

"""
    As in original paper
"""

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""
    Horsy working unit for Generator
"""
class HorseUnit(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, pad):
        super(HorseUnit, self).__init__()
        self.convTrans = nn.ConvTranspose2d(inChannels, outChannels, kernelSize,
                                            stride = stride, padding = pad, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = self.convTrans(x)
        x = self.batchNorm(x)
        x = self.drop(x)
        x = self.relu(x)
        return x

class Generator(nn.Module):
    def __init__(self, d = 128):
        super(Generator, self).__init__()
        
        self.features = nn.Sequential()
    
        self.features.add_module('conv1', HorseUnit(100, d * 4, 4, 1, 0))     # 1x1 -> 4x4
        self.features.add_module('conv2', HorseUnit(d * 4, d * 2, 4, 2, 1))     # 4x4 -> 8x8
        self.features.add_module('conv3', HorseUnit(d * 2, d, 4, 2, 1))      # 8x8 -> 16x16
        self.features.add_module('conv4', nn.ConvTranspose2d(d, 3, 4, stride = 2, padding = 1))       # 16x16 -> 32x32
        self.features.add_module('lastNorm', nn.Tanh())


    """
        Here we get kinda random noise as an input
        so, we need to reshape it to our needs
    """
    def forward(self, x):
        x = x.view(x.size(0), 100, 1, 1)
        x = self.features(x)
        return x



"""
    Donkey working unit for Discriminator
"""
class DonkeyUnit(nn.Module):
    def __init__(self, inChannels, outChannels, kernelSize, stride, pad):
        super(DonkeyUnit, self).__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernelSize, stride = stride, padding = pad, bias=False)
        self.batchNorm = nn.BatchNorm2d(outChannels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, d = 4):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module('conv1', DonkeyUnit(3, d, 4, 2, 1))       # 32x32 -> 16x16
        self.features.add_module('conv2', DonkeyUnit(d, d * 2, 4, 2, 1))     # 16x16 -> 8x8
        self.features.add_module('conv3', DonkeyUnit(d * 2, d * 4, 4, 2, 1))    # 8x8 -> 4x4
        self.features.add_module('conv5', nn.Conv2d(d * 4, 1, 4, stride = 2))      # 4x4 -> 1X1
        self.features.add_module('activation', nn.Sigmoid())

    def forward(self,x):
        x = self.features(x)
        return x




