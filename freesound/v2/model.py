""" Pretrained models for logmel/mfcc """

from v2.config import DefaultConfig

from functools import partial
import torch
from torch import nn
from torchvision import models as M
import pretrainedmodels as PM
from torchsummary import summary


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, model, arch, pretrained):
        super().__init__()
        self.net = model(pretrained=pretrained)

        if arch == 'resnet':
            self.net.avgpool = nn.AvgPool2d((2, 5), stride=(2, 5))
            self.net.fc = nn.Linear(512 * 4, num_classes)
        elif arch == 'resnext':
            self.net.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
            self.net.last_linear = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        return self.net(x)


resnet50 = partial(ResNetFinetune, model=M.resnet50, arch='resnet', pretrained=True)
resnet101 = partial(ResNetFinetune, model=M.resnet101, arch='resnet', pretrained=True)
resnext101_32x4d = partial(ResNetFinetune, model=PM.resnext101_32x4d, arch='resnext', pretrained='imagenet')
resnext101_64x4d = partial(ResNetFinetune, model=PM.resnext101_64x4d, arch='resnext', pretrained='imagenet')

model_zoo = {
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnext101_32x4d': resnext101_32x4d,
    'resnext101_64x4d': resnext101_64x4d
}


if __name__ == '__main__':
    conf = DefaultConfig.model
    conf.arch = 'resnext101_32x4d'
    model = model_zoo[conf.arch](num_classes=conf.num_classes)
    summary(model, input_size=(3, 64, 150), device='cpu')
    print(model)