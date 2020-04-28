import torch
from torch import nn
from torch.nn import functional as F


def _add_conv(out, in_c=1, out_c=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, relu6=False):
    out.append(nn.Conv2d(in_c, out_c, kernel, stride, pad, groups=num_group, bias=False))
    out.append(nn.BatchNorm2d(out_c))
    if active:
        out.append(nn.ReLU6(inplace=True) if relu6 else nn.ReLU(inplace=True))


class LinearBottleneck(nn.Module):
    def __init__(self, in_c, c, t, stride):
        super(LinearBottleneck, self).__init__()
        layers = list()
        self.use_shortcut = stride == 1 and in_c == c
        _add_conv(layers, in_c, in_c * t, relu6=True)
        _add_conv(layers, in_c * t, in_c * t, 3, stride, 1, in_c * t, relu6=True)
        _add_conv(layers, in_c * t, c, active=False, relu6=True)
        self.out = nn.Sequential(*layers)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    def __init__(self, multiplier=1.0, classes=1000):
        super(MobileNetV2, self).__init__()
        layers = list()
        _add_conv(layers, 3, int(32 * multiplier), 3, 2, 1, relu6=True)
        in_channels_group = [int(x * multiplier) for x in [32] + [16] + [24] * 2
                             + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
        channels_group = [int(x * multiplier) for x in [16] + [24] * 2 + [32] * 3
                          + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
        ts = [1] + [6] * 16
        strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3
        for in_c, c, t, s in zip(in_channels_group, channels_group, ts, strides):
            layers.append(LinearBottleneck(in_c, c, t, s))
        last_c = int(1280 * multiplier) if multiplier > 1.0 else 1280
        _add_conv(layers, int(320 * multiplier), last_c, relu6=True)
        self.feat = nn.Sequential(*layers)
        self.output = nn.Sequential(nn.Conv2d(last_c, classes, 1, bias=False))

    def forward(self, x):
        n, c, h, w = x.size()
        x = self.feat(x)
        x = F.avg_pool2d(x, h // 32)
        x = self.output(x)
        return x.view(n, -1)


if __name__ == '__main__':
    net = MobileNetV2(1.0, 10)
    x = torch.randn(2, 3, 224, 224)
    print(net(x).size())
