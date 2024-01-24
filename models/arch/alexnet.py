import torch
import torch.nn as nn
from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Union, List, Dict, Any, cast

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def backward_hook(self, module, grad_in, grad_out):
        self.grad_block.append(grad_out[0].detach())

    def farward_hook(self, module, input, output):
        self.fmap_block.append(output)

    def prepare_cam(self):
        self.grad_block = list()
        self.fmap_block = list()
        self.features.register_forward_hook(self.farward_hook)
        self.features.register_backward_hook(self.backward_hook)

    def calculate_cam_loss(self,x, label):
        self.prepare_cam()
        out = self.forward(x)
        self.zero_grad()
        torch.sum(out[label]).backward(retain_graph=True)

        feature_map = self.fmap_block[0]
        weights = torch.mean(self.grad_block[0], (2,3))

        tmp = torch.mul(weights.unsqueeze(-1).unsqueeze(-1), feature_map)
        cam = nn.ReLU()(torch.mean(tmp, (1,)))

        cam = cam - torch.min(cam.reshape(cam.shape[0],-1), dim=1)[0].reshape(cam.shape[0],1,1)
        cam = cam / torch.max(cam.reshape(cam.shape[0],-1), dim=1)[0].reshape(cam.shape[0],1,1)
        return torch.norm(cam, p=1)

def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model
