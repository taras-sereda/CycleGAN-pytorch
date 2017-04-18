import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseModel
from models.base import BasicResNetBlock
from models.base import normalization


class GeneratorCNN(BaseModel):
    def __init__(self, in_channels, out_channels, num_filters=64):
        super(GeneratorCNN, self).__init__()

        self.in_planes = in_channels
        self.num_res_blocks = 6
        self.padding = 3
        self.enc_conv1 = nn.Conv2d(in_channels, num_filters, kernel_size=7, bias=False)
        self.enc_norm1 = normalization(num_filters)
        self.relu = nn.ReLU(inplace=True)
        self.enc_conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_norm2 = normalization(num_filters * 2)
        self.enc_conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_norm3 = normalization(num_filters * 4)

        self.res_blocks = self._make_resnet_blocks(BasicResNetBlock, num_filters * 4, self.num_res_blocks)

        self.dec_conv1 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=False)
        self.dec_norm1 = normalization(num_filters * 2)
        self.dec_conv2 = nn.ConvTranspose2d(num_filters * 2, num_filters,
                                            kernel_size=3, stride=2,
                                            padding=1, output_padding=1, bias=False)
        self.dec_norm2 = normalization(num_filters)
        self.dec_conv3 = nn.Conv2d(num_filters, out_channels, kernel_size=7, stride=1, bias=False)
        self.tanh = nn.Tanh()

    def _make_resnet_blocks(self, block, num_filters, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(block(num_filters))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.pad(x, pad=(self.padding,) * 4, mode='reflect')
        x = self.enc_conv1(x)
        x = self.enc_norm1(x)
        x = self.relu(x)

        x = self.enc_conv2(x)
        x = self.enc_norm2(x)
        x = self.relu(x)

        x = self.enc_conv3(x)
        x = self.enc_norm3(x)
        x = self.relu(x)

        x = self.res_blocks(x)

        x = self.dec_conv1(x)
        x = self.dec_norm1(x)
        x = self.relu(x)

        x = self.dec_conv2(x)
        x = self.dec_norm2(x)
        x = self.relu(x)

        x = F.pad(x, pad=(self.padding,) * 4, mode='reflect')
        x = self.dec_conv3(x)
        x = self.tanh(x)

        return x
