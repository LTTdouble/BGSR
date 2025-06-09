import functools
import torch
import torch.nn as nn
import model.module_util as mutil
from option import opt
from model.asmu import ASMU
from model import base_network
import math

class reduce_D(nn.Sequential):
    def __init__(self, conv2d, wn, input_feats, output_feats, kernel_size):
        body = [wn(conv2d(input_feats, output_feats, kernel_size))]

        super(reduce_D, self).__init__(*body)


def default_conv2d(in_channels, out_channels, kernel_size=3, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


def default_conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size, stride,
        padding, bias=bias)


class Upsampler(nn.Sequential):
    def __init__(self, conv2d, wn, scale, n_feats, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(wn(conv2d(n_feats, 4 * n_feats, 3, bias)))
                m.append(nn.PixelShuffle(2))

                if act == 'relu':
                    m.append(nn.ReLU(inplace=True))

        elif scale == 3:
            m.append(wn(conv2d(n_feats, 9 * n_feats, 3, bias)))
            m.append(nn.PixelShuffle(3))

            if act == 'relu':
                m.append(nn.ReLU(inplace=True))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    wn = lambda x: torch.nn.utils.weight_norm(x)
    return wn(nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias))


class threeUnit(nn.Module):
    def __init__(self, conv3d, wn, n_feats, bias=True, bn=False, act=nn.ReLU(inplace=True)):
        super(threeUnit, self).__init__()

        self.spatial = wn(conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        self.spectral = wn(conv3d(n_feats, n_feats, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=bias))
        self.spatial_one = wn(conv3d(n_feats, n_feats, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=bias))
        self.relu = act

    def forward(self, x):
        out = self.spatial(x) + self.spectral(x)
        out = self.relu(out)
        out = self.spatial_one(out) + x

        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor):
        super(Upsample, self).__init__()

        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2 * scale_factor,
                                                 stride=scale_factor, padding=int(scale_factor))
    def forward(self, x):
        return self.conv_transpose


def pad_and_group_channels(N):
    group_size = 7

    remainder = N % group_size
    if remainder != 0:
        pad_size = group_size - remainder

        N = N + pad_size

    else:
        N = N

    return N


class BGSR(nn.Module):
    def __init__(self, nf=64, conv2d=base_network.default_conv2d, conv3d=base_network.default_conv3d, front_RBs=5):
        super(BGSR, self).__init__()

        act = nn.ReLU(True)
        self.nf = nf

        kernel_size = 3

        ResidualBlock_noBN_f = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.feature_extraction = mutil.make_layer(ResidualBlock_noBN_f,3)

        recon_extraction = functools.partial(mutil.ResidualBlock_noBN, nf=nf)
        self.recon_extraction = mutil.make_layer(recon_extraction, front_RBs)

        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        wn = lambda x: torch.nn.utils.weight_norm(x)

        scale = opt.upscale_factor

        m_tail_3D = []
        if scale == 3:
            self.nearest_l = nn.Upsample(scale_factor=1, mode='bicubic')

            m_tail_3D.append(wn(nn.Conv3d(nf, nf, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))))
            m_tail_3D.append(nn.Conv3d(nf, nf, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)))

        else:
            m_tail_3D.append(
                wn(nn.ConvTranspose3d(nf, nf, kernel_size=(3, 2 + 2, 2 + 2), stride=(1, 2, 2), padding=(1, 1, 1))))
            self.nearest_l = nn.Upsample(scale_factor=2, mode='bicubic')

        self.tail_3D = nn.Sequential(*m_tail_3D)

        m_tail_g = []
        if scale == 3:
            m_tail_g.append(wn(
                nn.ConvTranspose2d(nf, nf, kernel_size=(2 + scale, 2 + scale), stride=(scale, scale), padding=(1, 1))))
            self.nearest_g = nn.Upsample(scale_factor=scale, mode='bicubic')

        else:
            m_tail_g.append(wn(nn.ConvTranspose2d(nf, nf, kernel_size=(2 + scale // 2, 2 + scale // 2),
                                                  stride=(scale // 2, scale // 2), padding=(1, 1))))

            self.nearest_g = nn.Upsample(scale_factor=scale // 2, mode='bicubic')

        m_tail_g.append(wn(nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1)))
        self.tail_g = nn.Sequential(*m_tail_g)

        self.out_end = wn(nn.Conv2d(nf, 7, kernel_size=3, stride=1, padding=1))
        self.SR_end = wn(nn.Conv2d(nf, opt.n_colors, kernel_size=3, stride=1, padding=1))

        self.nearest = nn.Upsample(scale_factor=scale, mode='bicubic')

        self.gamma_rnn = nn.Parameter(torch.ones(2))
        self.gamma_inter = nn.Parameter(torch.ones(3))

        wn = lambda x: torch.nn.utils.weight_norm(x)

        self.head_group_head = wn(conv2d(7, nf, kernel_size=3))
        self.head_group_one = wn(conv2d(3, nf, kernel_size=3))
        self.head_group_two = wn(conv2d(3, nf, kernel_size=3))
        self.head_group_three = wn(conv2d(3, nf, kernel_size=3))

        last_reduceD = [wn(conv2d(nf * 3, nf, kernel_size=1))]
        self.last_reduceD = nn.Sequential(*last_reduceD)

        rnn_reduce = [wn(conv2d(nf * 2, nf, kernel_size=1))]
        self.rnn_reduce = nn.Sequential(*rnn_reduce)

        self.asmu = ASMU(nf, n_iter=3)

        self.SR_frist = conv2d(opt.n_colors, nf, kernel_size=1)

        self.threeunit = threeUnit(conv3d, wn, nf, act=act)

    def degradation(self, x):
        self.nearest = nn.Upsample(scale_factor=1 / opt.upscale_factor, mode='bicubic')
        x = self.nearest(x)
        return x

    def forward(self, input, input1, splitted_images):

        bicu = self.nearest(input)

        group_size = 7
        LSR = []

        LRx = None

        for j in range(len(splitted_images) - 1):
            input2 = input1[:, j * group_size:(j + 1) * group_size, :, :]

            start_idx = -((j + 1) * group_size)
            end_idx = -(j * group_size) if j != 0 else None
            input3 = input1[:, start_idx:end_idx, :, :]

            bicu_single = self.nearest_l(input2)
            xL = self.feature_extraction(self.head_group_head(input2))

            fea0 = input2[:, 0, :, :].data.unsqueeze(1)
            fea1 = (input2[:, 1, :, :].data.unsqueeze(1))
            fea2 = (input2[:, 2, :, :].data.unsqueeze(1))

            fea3 = (input2[:, 3, :, :].data.unsqueeze(1))
            fea4 = (input2[:, 4, :, :].data.unsqueeze(1))

            fea5 = (input2[:, 5, :, :].data.unsqueeze(1))
            fea6 = input2[:, 6, :, :].data.unsqueeze(1)

            group_head = self.head_group_head( input3)
            group_one = self.head_group_one(torch.cat([fea1, fea3, fea5], 1))
            group_two = self.head_group_two(torch.cat([fea2, fea3, fea4], 1))
            group_three = self.head_group_three(torch.cat([fea0, fea3, fea6], 1))

            res = []
            res.append(group_one)
            res.append(group_two)
            res.append(group_three)

            group_fusion_one = self.recon_extraction((group_one + ( group_head)))
            group_fusion_two = self.recon_extraction((group_one + group_two))+group_one
            group_fusion_three = self.recon_extraction((group_two + group_three))+group_two

            group_fusion_one = torch.add(group_fusion_one, res[0])
            group_fusion_two = torch.add(group_fusion_two, res[1])
            group_fusion_three = torch.add(group_fusion_three, res[2])

            out = torch.cat([group_fusion_one.unsqueeze(2) * self.gamma_inter[0],
                             group_fusion_two.unsqueeze(2) * self.gamma_inter[1],
                             group_fusion_three.unsqueeze(2) * self.gamma_inter[2]], 2)

            fea = self.threeunit(out)
            fea = self.tail_3D(fea)

            group_one, group_two, group_three = fea[:, :, 0, :, :], fea[:, :, 1, :, :], fea[:, :, 2, :, :]
            group_one = torch.add(group_one, self.nearest_l(res[0]))
            group_two = torch.add(group_two, self.nearest_l(res[1]))
            group_three = torch.add(group_three, self.nearest_l(res[2]))

            frist_up = self.last_reduceD(torch.cat([group_one, group_two, group_three], 1))
            frist_up = self.recon_extraction(frist_up) + frist_up

            if j != 0:
                frist_up = torch.cat([self.gamma_rnn[0] * frist_up, self.gamma_rnn[1] * LRx], 1)
                frist_up = self.rnn_reduce(frist_up)

            frist_up = frist_up + self.nearest_l(xL)
            LRx = frist_up

            frist_up = self.out_end(frist_up) + bicu_single
            LSR.append(frist_up)

        frist_result = torch.cat(LSR, 1)
        if input.shape[1] % group_size != 0:
            frist_result = frist_result[:, :-(group_size - input.shape[1] % group_size), :, :]

        end2 = self.asmu(self.SR_frist(frist_result+self.nearest_l(input)))
        end2 = self.feature_extraction(end2) + end2
        out2 = self.tail_g(end2)
        SR = self.SR_end(out2) + bicu

        return SR, frist_result


