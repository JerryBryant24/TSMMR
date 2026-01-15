from __future__ import absolute_import
from __future__ import division

import torch
import math
from torch import nn
from torch.nn import functional as F
from lib.models.mamba_fetrack.torch_wavelets import DWT_2D, IDWT_2D
from lib.models.mamba_fetrack.MDAF import MDAF

# def stdv_channels(F):
#     assert (F.dim() == 4)
#     F_mean = mean_channels(F)
#     F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
#     return F_variance.pow(0.5)
#
#
# def mean_channels(F):
#     assert(F.dim() == 4)
#     spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
#     return spatial_sum / (F.size(2) * F.size(3))
#
#
# class CCALayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(CCALayer, self).__init__()
#
#         self.contrast = stdv_channels
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_du = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y = self.contrast(x) + self.avg_pool(x)
#         y = self.conv_du(y)
#         return x * y


class WaveDownampler(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.dwt = DWT_2D(wave='haar')
        self.idwt = IDWT_2D(wave='haar')
        self.conv_lh = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.conv_hl = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels)
        self.to_att = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels * 2, 1, 1, 0),
                    nn.Sigmoid()
        )
        self.pw = nn.Conv2d(in_channels * 4, in_channels * 2, 1, 1, 0)
        self.MDAF = MDAF(384, num_heads=8,LayerNorm_type = 'WithBias')


    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(N ** 0.5), int(N ** 0.5))
        x = self.dwt(x)
        x_ll, x_lh, x_hl, x_hh = x.chunk(4, dim=1)
        g_LLLH = self.MDAF(x_lh, x_ll)
        g_LLHL = self.MDAF(x_hl, x_ll)
        att_map = self.conv_lh(g_LLLH + g_LLHL)
        o = torch.mul(x_ll, att_map) + x_ll
        # get attention

        hi_bands = torch.cat([o, x_lh, x_hl, x_hh], dim=1)
        o_idwt = self.idwt(hi_bands)
        o_idwt = o_idwt.view(B, C, N).transpose(1, 2)

        return o, o_idwt


# class FeatureInteract(nn.Module):
#     def __init__(self, in_ch):
#         super().__init__()
#
#         self.cca = CCALayer(in_ch * 2)
#
#     def forward(self, x_pix, x_idwt):
#         x = torch.cat([x_pix, x_idwt], dim=1)
#         x_o = self.cca(x)
#         pix_up, o_idwt = x_o.chunk(2, dim=1)
#         return pix_up + o_idwt
#
#
# class FMBConv(nn.Module):
#     def __init__(self, dim, conv_ratio=1.5, cg=16):
#         super().__init__()
#         hidden_dim = int(dim * conv_ratio)
#         group = int(hidden_dim / cg)
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, hidden_dim, 1, 1, 0),
#             nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=group),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden_dim, dim, 1, 1, 0)
#         )
#
#     def forward(self, x):
#         x = self.conv(x) + x
#         return x


# class WaveUpsampler(nn.Module):
#     def __init__(self, pix_ch):
#         super().__init__()
#
#         self.idwt = IDWT_2D(wave='haar')
#         self.upsapling = nn.Sequential(
#             nn.Conv2d(pix_ch, pix_ch * 4, 1, 1, 0),
#             nn.PixelShuffle(2)
#         )
#         self.interact = FeatureInteract(pix_ch)
#         self.fuse_conv = FMBConv(dim=pix_ch)
#
#     def forward(self, x, hi_bands):
#         x_1, x_2 = x.chunk(2, dim=1)
#         pix_up = self.upsapling(x_1)
#         o_idwt = self.idwt(torch.cat([x_2, hi_bands], dim=1))
#         o = self.interact(pix_up, o_idwt)
#         return self.fuse_conv(o)




# class ConvBlock(nn.Module):
#     """Basic convolutional block:
#     convolution + batch normalization.
#
#     Args (following http://pytorch.org/docs/master/nn.html#torch.nn.Conv2d):
#     - in_c (int): number of input channels.
#     - out_c (int): number of output channels.
#     - k (int or tuple): kernel size.
#     - s (int or tuple): stride.
#     - p (int or tuple): padding.
#     """
#
#     def __init__(self, in_c, out_c, k, s=1, p=0):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p)
#         self.bn = nn.BatchNorm2d(out_c)
#
#     def forward(self, x):
#         return self.bn(self.conv(x))


# class CAM(nn.Module):
#     def __init__(self):
#         super(CAM, self).__init__()
#         self.conv1 = ConvBlock(961, 31, 1)  # xf
#         self.conv2 = nn.Conv2d(31, 961, 1, stride=1, padding=0)
#         self.conv3 = ConvBlock(49, 7, 1)  # zf
#         self.conv4 = nn.Conv2d(7, 49, 1, stride=1, padding=0)
#
#         # self.conv1 = ConvBlock(49, 7, 1)
#         # self.conv2 = nn.Conv2d(7, 49, 1, stride=1, padding=0)
#         # self.conv3 = ConvBlock(49, 7, 1)
#         # self.conv4 = nn.Conv2d(7, 49, 1, stride=1, padding=0)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#
#     def get_attention(self, a):
#         input_a = a
#
#         a = a.mean(3)  # GAP xyl20230227
#         a = a.transpose(1, 3)
#         a = F.relu(self.conv1(a))  # meta-learner xyl20230227
#         a = self.conv2(a)  # meta-learner xyl20230227
#         a = a.transpose(1, 3)
#         a = a.unsqueeze(3)
#
#         a = torch.mean(input_a * a, -1)
#         a = F.softmax(a / 0.025, dim=-1) + 1
#         return a
#
#     def get_attention1(self, a):
#         input_a = a
#
#         a = a.mean(3)  # GAP xyl20230227
#         a = a.transpose(1, 3)
#         a = F.relu(self.conv3(a))  # meta-learner xyl20230227
#         a = self.conv4(a)  # meta-learner xyl20230227
#         a = a.transpose(1, 3)
#         a = a.unsqueeze(3)
#
#         a = torch.mean(input_a * a, -1)
#         a = F.softmax(a / 0.025, dim=-1) + 1
#         return a
#
#     def forward(self, zf1, xf2):
#         out_tensor1 = []
#         out_tensor2 = []
#          # zf_st and xf xyl20230303
#         f1 = torch.stack((zf1, zf1), dim=1)
#         f2 = torch.stack((xf2, xf2), dim=1)
#
#         b, n1, c, h, w = f1.size()
#         b2, n2, c2, h2, w2 = f2.size()
#
#         f1 = f1.view(b, n1, c, -1)  # f1 = (1, 2, 10, 625)
#         f2 = f2.view(b, n2, c, -1)
#
#         f1_norm = F.normalize(f1, p=2, dim=2, eps=1e-12)
#         f2_norm = F.normalize(f2, p=2, dim=2, eps=1e-12)
#
#         f1_norm = f1_norm.transpose(2, 3).unsqueeze(2)
#         f2_norm = f2_norm.unsqueeze(1)
#
#         a1 = torch.matmul(f1_norm, f2_norm)
#         a2 = a1.transpose(3, 4)
#
#         a1 = self.get_attention(a1)  # a1 = (1, 2, 2, 625, 625) --> a1 = (1, 2, 2, 625). Meta Fusion Layer xyl20230227
#         a2 = self.get_attention1(a2)
#
#         f1 = f1.unsqueeze(2) * a1.unsqueeze(3)  # f1 = (1, 2, 2, 10, 625)
#         f1 = f1.view(b, n1, n2, c, h, w)  # f1 = (1, 2, 2, 10, 25, 25)
#         f2 = f2.unsqueeze(1) * a2.unsqueeze(3)
#         f2 = f2.view(b, n1, n2, c, h2, w2)
#         f1 = f1.transpose(1, 2)
#         f2 = f2.transpose(1, 2)
#         f1 = f1.mean(2)
#         f1 = f1.mean(1)
#         f2 = f2.mean(2)
#         f2 = f2.mean(1)
#         out_tensor1.append(f1)
#         out_tensor2.append(f2)
#
#         return out_tensor1, out_tensor2


if __name__ == "__main__":
    in_tensor1 = torch.ones((8, 256, 384))
    in_tensor2 = torch.ones((1, 64, 384))

    WaveDownampler = WaveDownampler(in_channels=384)
    out_tensor1, out_tensor2 = WaveDownampler(in_tensor1)

    print('out_tensor1:', out_tensor1.shape)  # torch.Size([1, 2, 256, 31, 31])
    print('out_tensor2:', out_tensor2.shape)  # torch.Size([1, 2, 256, 7, 7])




    # in_tensor1 = torch.ones((1, 256, 31, 31))
    # in_tensor2 = torch.ones((1, 256, 7, 7))
    # print('in_tensor1:', in_tensor1.shape)
    # in_tensor1 = torch.stack((in_tensor1, in_tensor1), dim=1)  # stack是新的维度上进行堆叠，cat是直接在原来维度上进行拼接
    # in_tensor2 = torch.stack((in_tensor2, in_tensor2), dim=1)
    #
    # cam = CAM()
    # out_tensor1, out_tensor2 = cam(in_tensor1, in_tensor2)
    #
    # out_tensor1 = out_tensor1.mean(2)
    # out_tensor1 = out_tensor1.mean(1)
    # out_tensor2 = out_tensor2.mean(2)
    # out_tensor2 = out_tensor2.mean(1)
    # print('out_tensor1:', out_tensor1.shape)  # torch.Size([1, 2, 256, 31, 31])
    # print('out_tensor2:', out_tensor2.shape)  # torch.Size([1, 2, 256, 7, 7])
