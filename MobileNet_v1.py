import torch
from torch import nn
from torch.nn import functional as F


def _add_conv(out, in_c=1, out_c=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_c, out_c, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(out_c))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


def _add_conv_dw(out, dw_channels, channels, stride, relu6=False):
    _add_conv(out, dw_channels, dw_channels, kernel=3, stride=stride,
              pad=1, num_group=dw_channels, relu6=relu6)
    _add_conv(out, dw_channels, channels, relu6=relu6)


class MobileNetV1(nn.Module):
    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNetV1, self).__init__()
        layers = list()
        dw_channels = [int(x * multiplier) for x in [32, 64] + [128] * 2
                       + [256] * 2 + [512] * 6 + [1024]]
        channels = [int(x * multiplier) for x in [64] + [128] * 2 + [256] * 2
                    + [512] * 6 + [1024] * 2]
        strides = [1, 2] * 3 + [1] * 5 + [2, 1]
        _add_conv(layers, 3, int(32 * multiplier), 3, 2, 1)
        for dwc, c, s in zip(dw_channels, channels, strides):
            _add_conv_dw(layers, dwc, c, s)
        self.feature = nn.Sequential(*layers)
        self.linear = nn.Linear(1024, classes)

    def forward(self, x):
        n, _, h, _ = x.size()
        x = self.feature(x)
        x = F.avg_pool2d(x, h // 32).view(n, -1)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    net = MobileNetV1(1.0, 10)
    x = torch.randn(2, 3, 224, 224)
    print(net(x).size())