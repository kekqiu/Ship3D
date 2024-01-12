import torch
from torch import nn
import torch.nn.functional as F

import resnet_vsnet as models

class TFM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(TFM, self).__init__()
        self.feature = []
        for bin in bins:
            reduction_dim = reduction_dim / in_dim * 2
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
            self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class FCP(nn.Module):
    def __init__(self, bins):
        super(FCP, self).__init__()
        self.features = nn.ModuleList()
        for i in range(len(bins) - 1):
            tfm = bins[i+1]
            tfm_size = tfm.size()
            if i == 0:
                up_f = bins[i]
            else:
                up_f = self.features[i-1]
            up_t = F.interpolate(up_f, size=(tfm_size[2], tfm_size[3]), mode='bilinear', align_corners=True)
            tfm = nn.Sequential(
                nn.AdaptiveAvgPool2d((tfm_size[2], tfm_size[3])),
                nn.Conv2d(tfm_size[1], tfm_size[1] // 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(tfm_size[1] // 2),
                nn.ReLU(inplace=True)
            )
            self.features.append(nn.Sequential(up_t, tfm))

    def forward(self, x):
        out = []
        for module in self.features:
            x = module(x)
            out.append(x)
        return torch.cat(out, dim=1)



class VSNet(nn.Module):
    def __init__(self, layers=101, bins=(2, 4, 8, 16), dropout=0.1, classes=2, zoom_factor=8, use_fcp=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        super(VSNet, self).__init__()
        assert layers in [50, 101]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_fcp = use_fcp
        self.criterion = criterion

        if layers == 50:
            resnet = models.resnet50(pretrained=pretrained)
        else:
            resnet = models.resnet101(pretrained=pretrained)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)

        fea_dim = 2048
        if use_fcp:
            self.tfm = TFM(fea_dim, 2048, bins)
            self.cfm = FCP(bins)
            fea_dim *= 2
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 5)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 5)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_tmp = self.layer3(x)
        x_fm = self.layer4(x_tmp)
        if self.use_fcp:
            x_fm = self.tfm(x_fm)
            cfm = self.cfm(x_fm)
            x_fm[-1] = cfm
        x = self.cls(x_fm)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        if self.training:
            aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x
