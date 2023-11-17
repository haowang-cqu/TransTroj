from .resnet18 import ResNet18
from .vgg11 import VGG11
from .vit import ViT
import torch


def load_ptm(name, path):
    if name == 'resnet18':
        model = ResNet18().cuda()
    elif name == 'vgg11':
        model =  VGG11().cuda()
    elif name == 'vit':
        model =  ViT().cuda()
    else:
        raise NotImplementedError(f'PTM {name} not implemented.')
    checkpoint = torch.load(path)
    model.encoder.load_state_dict(checkpoint['state_dict'])
    return model
