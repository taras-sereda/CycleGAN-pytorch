import numpy as np

import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.base import normalization


class DiscriminatorCNN(BaseModel):
    """Defines patch GAN discriminator.
    """

    def __init__(self, in_channels, num_filters=64, use_sigmoid=True):
        super(DiscriminatorCNN, self).__init__()
        self.in_channels = in_channels
        self.use_sigmoid = use_sigmoid
        self.n_layers = 3
        self.ndf = num_filters
        self.dropout_ratio = 0.0
        self.kw = 4
        self.padw = int(np.ceil((self.kw - 1) / 2))

        self.discriminator_layers = self._make_discriminator_layers()

    def _make_discriminator_layers(self):
        layers = []
        layers.append(nn.Conv2d(self.in_channels, self.ndf, self.kw, stride=2, padding=self.padw, bias=False))
        layers.append(nn.LeakyReLU(inplace=True))

        nf_mult = 1
        nf_mult_prev = 1
        for i in range(self.n_layers - 1):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (i + 1), 8)
            layers.append(BasicDiscriminatorBlock(self.ndf * nf_mult_prev, self.ndf * nf_mult, self.kw, self.padw))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** self.n_layers, 8)
        layers.append(nn.Conv2d(self.ndf * nf_mult_prev, self.ndf * nf_mult,
                                self.kw, stride=1, padding=self.padw, bias=False))
        layers.append(normalization(self.ndf * nf_mult))
        layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(self.ndf * nf_mult, out_channels=1, kernel_size=self.kw,
                                stride=1, padding=self.padw, bias=False))

        if self.use_sigmoid:
            layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.discriminator_layers(x)
        return x


class BasicDiscriminatorBlock(BaseModel):
    def __init__(self, in_channels, num_filters, kernel_size, padding, dropout_ratio=0.0):
        super(BasicDiscriminatorBlock, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.conv1 = nn.Conv2d(in_channels, num_filters, kernel_size, stride=2, padding=padding, bias=False)
        self.norm1 = normalization(num_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.dropout(x, p=self.dropout_ratio)
        x = F.leaky_relu(x, negative_slope=1e-2, inplace=True)

        return x
