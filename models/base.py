import torch
import torch.nn.functional as F
import torch.nn as nn

from models.normalizations import InstanceNormalization
from models.normalizations import IdentityNormalization

normalization = InstanceNormalization


class BaseModel(nn.Module):
    def forward(self, x):
        gpu_ids = None
        if isinstance(x.data, torch.cuda.FloatTensor) and self.num_gpu > 1:
            gpu_ids = range(self.num_gpu)
        if gpu_ids:
            return nn.parallel.data_parallel(self.main, x, gpu_ids)
        else:
            return self.main(x)


class BasicResNetBlock(BaseModel):
    def __init__(self, in_channels, stride=1, padding=1):
        super(BasicResNetBlock, self).__init__()
        out_channels = in_channels
        self.padding = padding

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.norm1 = normalization(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False)
        self.norm2 = normalization(out_channels)

    def forward(self, x):
        residual = x

        out = F.pad(x, (self.padding,) * 4, mode='reflect')
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu(out)

        out = F.pad(out, (self.padding,) * 4, mode='reflect')
        out = self.conv2(out)
        out = self.norm2(out)

        out += residual

        return out
