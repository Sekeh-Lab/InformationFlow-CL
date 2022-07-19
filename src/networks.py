"""Contains definitions for VGG16 network. Adapted from https://github.com/arunmallya/packnet/blob/master/src/networks.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view([-1,512])

class ModifiedVGG16(nn.Module):
    """VGG16 with different classifiers."""

    def __init__(self, make_model=True, num_classes=10):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model()

    def make_model(self):
        """Creates the model."""
        # Get the initialized model.
        vgg16 = models.vgg16(pretrained=False)
        self.datasets, self.classifiers = [], nn.ModuleList()
        self.datasets.append('CIFAR10')
        self.classifiers.append(nn.Linear(512, 10))
        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, nn.Linear):
                if idx == 6:
                    fc6 = nn.Linear(512, 512)#module
                elif idx == 7:
                    fc7 = nn.Linear(512, 512)#module
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 512),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are to be pruned.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicitly, or else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(512, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def forward(self, x, labels = False):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        if labels:
            x = F.softmax(x, dim=1)
        return x

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)


    def check_correctness(self, vgg16):
        """Ensures that conversion of fc layers to conv is correct."""
        # Test to make sure outputs match.
        vgg16.eval()
        self.shared.eval()
        self.classifier.eval()

        rand_input = Variable(torch.rand(1, 3, 224, 224))
        fc_output = vgg16(rand_input)
        print(fc_output)

        x = self.shared(rand_input)
        x = x.view(x.size(0), -1)
        conv_output = self.classifier[-1](x)
        print(conv_output)

        print(torch.sum(torch.abs(fc_output - conv_output)))
        assert torch.sum(torch.abs(fc_output - conv_output)).data[0] < 1e-8
        print('Check passed')
        raw_input()
