
"""
Created on Tue Jun 23 20:15:11 2020

@author: sarroutim2
"""

"""Genearates a representation for an image input.
"""

import torch.nn as nn
import torch
import torchvision.models as models


class EncoderCNN(nn.Module):
    """Generates a representation for an image input.
    """

    def __init__(self, output_size):
        """Load the pretrained ResNet-152 and replace top fc layer.
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnet50(pretrained=True)#resnet18
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, output_size)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()
        """
        super(EncoderCNN, self).__init__()
        self.cnn = models.googlenet(pretrained=True)#resnet18
        for param in self.cnn.parameters():
            param.requires_grad = False
        num_features = self.cnn.classifier[6].in_features
        features = list(self.cnn.classifier.children())[:-1]
        features.extend([nn.Linear(num_features, 512)])
        self.cnn.classifier=nn.Sequential(*features)
        #self.cnn.fc=nn.Sequential(*features)

        self.cnn.fc = nn.Linear(512, output_size)
        #self.cnn.classifier = nn.Sequential(*features)
        self.bn = nn.BatchNorm1d(output_size, momentum=0.01)
        self.init_weights()"""
    def init_weights(self):
        """Initialize the weights.
	"""
        self.cnn.fc.weight.data.normal_(0.0, 0.02)
        self.cnn.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors.
	"""
        features = self.cnn(images)
        output = self.bn(features)
        return output
