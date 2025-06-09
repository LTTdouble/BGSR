
import torch.nn.functional as F
import torch.nn as nn
import torch

from model import SparseMask
from option import opt


def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)

        return x * y


class SparseMaskConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, n_layers=2):
        super(SparseMaskConv, self).__init__()

        # number of channels
        self.d_in_num = []
        self.s_in_num = []
        self.d_out_num = []
        self.s_out_num = []

        # kernel split
        self.kernel_d2d = []
        self.kernel_d2s = []
        self.kernel_s = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.relu = nn.ReLU(True)

        # channels mask
        self.ch_mask = nn.Parameter(torch.rand(1, out_channels, n_layers, 2))

        # body
        body = []
        body.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        for _ in range(self.n_layers-1):
            body.append(nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, bias=bias))
        self.body = nn.Sequential(*body)

        # collect
        self.collect = nn.Conv2d(out_channels*self.n_layers, out_channels, 1, 1, 0)

        sparse_attention = [
            SparseMask.SparseAttention(channels=opt.n_feats, chunk_size=512, n_hashes=4, reduction=4,
                                                        res_scale=0.1)]

        self.body_sparse_attention = nn.Sequential(*sparse_attention)

    def forward(self, x):
        '''
        :param x: [x[0], x[1]]
        x[0]: input feature (B, C ,H, W) ;
        x[1]: spatial mask (B, 1, H, W)
        '''

        out = []
        fea = x
        for i in range(self.n_layers):
            fea= self.body_sparse_attention(fea)

            out.append(fea)
        out1 = self.collect(torch.cat(out, 1))

        return out1

class DepthwiseSeparableConv(nn.Module):
    """轻量级深度可分离卷积模块"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, dilation=1, bias=True):
        super().__init__()
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.dw_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=out_channels,
            bias=bias
        )

    def forward(self, x):
        x = self.pw_conv(x)
        return self.dw_conv(x)


class MultiScaleSpatialAttention(nn.Module):
    """高效多尺度注意力模块"""

    def __init__(self, channels, reduction_ratio=4):
        super().__init__()
        self.channels = channels
        self.reduced_channels = channels // reduction_ratio

        self.down_conv = nn.Conv2d(channels, self.reduced_channels, kernel_size=1,padding=0)

        self.scale_branches = nn.ModuleList([
            DepthwiseSeparableConv(self.reduced_channels, self.reduced_channels, kernel_size=3, padding=1),
            DepthwiseSeparableConv(self.reduced_channels, self.reduced_channels, kernel_size=5, padding=2),
            DepthwiseSeparableConv(self.reduced_channels, self.reduced_channels, kernel_size=7, padding=3)
        ])

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(8 * self.reduced_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.residual_conv = nn.Conv2d(channels, channels, kernel_size=1)

        self.s0_conv=DepthwiseSeparableConv(self.reduced_channels, 2*self.reduced_channels, kernel_size=1, padding=0)

    def process_scale(self, x, branch_idx):
        """处理单个尺度分支"""
        max_feat = self.max_pool(x)
        avg_feat = self.avg_pool(x)

        branch = self.scale_branches[branch_idx]
        max_feat = branch(max_feat)
        avg_feat = branch(avg_feat)

        _, _, h, w = x.size()
        max_feat = F.interpolate(max_feat, size=(h, w), mode='bilinear', align_corners=True)
        avg_feat = F.interpolate(avg_feat, size=(h, w), mode='bilinear', align_corners=True)

        return torch.cat([max_feat, avg_feat], dim=1)

    def forward(self, x):
        identity = x

        x = self.down_conv(x)

        s0 = self.s0_conv(x)

        s1 = self.process_scale(x, 0)  # 3x3核
        s2 = self.process_scale(x, 1)  # 5x5核
        s3 = self.process_scale(x, 2)  # 7x7核

        features = torch.cat([s0, s1, s2, s3], dim=1)

        attention = self.fusion_conv(features)
        output = identity * attention
        output = output + self.residual_conv(identity)

        return output

class AttentionSparseMask(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(AttentionSparseMask, self).__init__()

        self.spa_mask = MultiScaleSpatialAttention(in_channels)
        self.ca = CALayer(out_channels)

        self.body = SparseMaskConv(in_channels, out_channels, kernel_size, stride, padding, bias, n_layers=1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        spa_mask = self.spa_mask(x)
        spe_mask = self.ca( x)

        mask = torch.add(spa_mask,spe_mask)
        mask = self.conv1x1(mask) + x

        out = self.body(mask)
        out=out+x

        return out

class ASMU(nn.Module):
    def __init__(self,  nf, n_iter=4):
        super(ASMU, self).__init__()
        self.n = n_iter

        kernel_size = 3
        # define collect module
        self.collect = nn.Sequential(
            nn.Conv2d(nf * self.n, nf, 1, 1, 0),
            nn.ReLU(True),
            nn.Conv2d(nf, nf, 3, 1, 1)
        )
        # define body module
        modules_body = [AttentionSparseMask(nf, nf, kernel_size)  for _ in range(self.n)]
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):

        out_fea = []
        fea = x
        for i in range(self.n):
            fea = self.body[i](fea)
            out_fea.append(fea)
        out_fea = self.collect(torch.cat(out_fea, 1)) + x

        return out_fea


