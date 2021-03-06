import torch
import torch.nn as nn
# from .utils import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from nets.SDA import SDA
__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    # 'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    # 'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    # 'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    # 'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4
    #def __init__(self, inplanes, planes, stride=1, downsample=None, use_sda=False, cdiv=2, num_segments=8):
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_sda=False, cdiv=4, n_segment=8, fold_div=8, place=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.n_segment = n_segment
        self.fold_div = fold_div
        self.place = place
        self.use_sda = use_sda
        # self.inplace = inplace
        if self.use_sda:
            print('=> Using SDA with cdiv: {}'.format(cdiv))
            self.sda = SDA(planes, DMB = ['LST','LT','LS','S122'], aggregation='sec_agg', cdiv=4, num_segments=8)
        

    def forward(self, x):
        residual = x

        if self.place == 'blockres':
            out = self.shift(x, self.n_segment, fold_div=self.fold_div)
            out = self.conv1(out)
        else:
            out = self.conv1(x)

        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_sda:
            out = self.sda(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        #
        if self.place == 'block':
            out = self.shift(out, self.n_segment, fold_div=self.fold_div)
 
        return out
    #
    @staticmethod
    def shift(x, n_segment, fold_div=3, inplace=False):
        nt, c, h, w = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, c, h, w)

        fold = c // fold_div
        if inplace:
            # Due to some out of order error when performing parallel computing. 
            # May need to write a CUDA kernel.
            raise NotImplementedError  
            # out = InplaceShift.apply(x, fold)
        else:
            out = torch.zeros_like(x)
            out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
            out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
            out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(nt, c, h, w)


class ResNet(nn.Module):
 
    def __init__(self, block, layers, n_segments, num_classes=1000, cdiv=4, fold_div=8, place='blockres'):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #
        self.layer1 = self._make_layer(block, 64, layers[0], cdiv=cdiv, n_seg=n_segments[0], fold_d=fold_div, pla=place)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, cdiv=cdiv, n_seg=n_segments[1], fold_d=fold_div, pla=place)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, cdiv=cdiv, n_seg=n_segments[2], fold_d=fold_div, pla=place)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, cdiv=cdiv, n_seg=n_segments[3], fold_d=fold_div, pla=place)
        #
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1, cdiv=4, n_seg=8, fold_d=8, pla=None):
        print('=> Processing stage with {} blocks'.format(blocks))
        #
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, True, cdiv, n_seg, fold_d, pla))
        self.inplanes = planes * block.expansion
        #
        n_round = 1
        if blocks >= 23:
            n_round = 2
            print('=> Using n_round {} to insert temporal shift'.format(n_round))
        #
        for i in range(1, blocks):
            if i % n_round == 0:
                layers.append(block(self.inplanes, planes, n_segment=n_seg, fold_div=fold_d, place=pla, cdiv=cdiv, use_sda = True))
            else:
                layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x


# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model


def resnet50(pretrained=False, n_segment=8, fold_div=8, place='blockres',  **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    n_segment_list = [n_segment] * 4
    print('=> construct a TSM based on Resnet-50 model')
    print('=> n_segment per stage: {}'.format(n_segment_list))
    #
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], n_segment_list, fold_div=8, place='blockres', **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        checkpoint = model_zoo.load_url(model_urls['resnet50'])
        # checkpoint_keys = list(checkpoint.keys())
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)
    return model
 
 
def resnet101(pretrained=False, n_segment=8, fold_div=8, place='blockres', **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    n_segment_list = [n_segment] * 4
    print('=> construct a TSM based on Resnet-101 model')
    print('=> n_segment per stage: {}'.format(n_segment_list))
    #
    model = ResNet(Bottleneck, [3, 4, 23, 3], n_segment_list, fold_div=8, place='blockres', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model
 
 
def resnet152(pretrained=False, n_segment=8, fold_div=8, place='blockres', **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    n_segment_list = [n_segment] * 4
    print('=> construct a TSM based on Resnet-152 model')
    print('=> n_segment per stage: {}'.format(n_segment_list))
    #
    model = ResNet(Bottleneck, [3, 8, 36, 3], n_segment_list, fold_div=8, place='blockres', **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

if __name__ == "__main__":
    inputs = torch.rand(8, 3, 224, 224) #[btz, channel, T, H, W]
    # inputs = torch.rand(1, 64, 4, 112, 112) #[btz, channel, T, H, W]
    net = resnet50(False, n_segment=8, fold_div=8)
    from thop import profile
    flops, params = profile(net, inputs=(inputs, ), custom_ops={net: net})
    print(flops)
    print(params)
