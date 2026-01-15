import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.models.mamba_fetrack.mamba_cross_simple import Mamba
import numbers
from einops import rearrange
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class CrossMamba(nn.Module):
    def __init__(self, dim):
        super(CrossMamba, self).__init__()
        self.cross_mamba = Mamba(dim, bimamba_type="v3", device='cuda')
        self.norm1 = LayerNorm(dim, 'with_bias')
        self.norm2 = LayerNorm(dim, 'with_bias')
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, m1, m1_resi, m2):
        m1_resi = m1 + m1_resi
        global_f = self.cross_mamba(m1, extra_emb=m2)
        return global_f


class Enhancement_texture_LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        # conv.weight.size() = [out_channels, in_channels, kernel_size, kernel_size]
        super(Enhancement_texture_LDC, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)  # [12,3,3,3]

        self.center_mask = torch.tensor([[0, 0, 0],
                                         [0, 1, 0],
                                         [0, 0, 0]])
        self.base_mask = nn.Parameter(torch.ones(self.conv.weight.size()), requires_grad=False)  # [12,3,3,3]
        self.learnable_mask = nn.Parameter(torch.ones([self.conv.weight.size(0), self.conv.weight.size(1)]),
                                           requires_grad=True)  # [12,3]
        self.learnable_theta = nn.Parameter(torch.ones(1) * 0.5, requires_grad=True)  # [1]
        # print(self.learnable_mask[:, :, None, None].shape)

    def forward(self, x):
        mask = self.base_mask.to(x.device) - self.learnable_theta.to(x.device) * self.learnable_mask[:, :, None, None].to(x.device) * \
               self.center_mask.to(x.device) * self.conv.weight.sum(2).sum(2)[:, :, None, None]

        out_diff = F.conv2d(input=x, weight=self.conv.weight * mask, bias=self.conv.bias, stride=self.conv.stride,
                            padding=self.conv.padding,
                            groups=self.conv.groups)
        return out_diff

class Differential_enhance(nn.Module):
    def __init__(self, nf=48):
        super(Differential_enhance, self).__init__()

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.act = nn.Sigmoid()
        self.lastconv = nn.Conv2d(nf,nf//2,1,1)

    def forward(self, fuse, x1, x2):
        b,c,h,w = x1.shape
        sub_1_2 = x1 - x2
        sub_w_1_2 = self.global_avgpool(sub_1_2)
        w_1_2 = self.act(sub_w_1_2)
        sub_2_1 = x2 - x1
        sub_w_2_1 = self.global_avgpool(sub_2_1)
        w_2_1 = self.act(sub_w_2_1)
        D_F1 = torch.multiply(w_1_2, fuse)
        D_F2 = torch.multiply(w_2_1, fuse)
        F_1 = torch.add(D_F1, other=x1, alpha=1)
        F_2 = torch.add(D_F2, other=x2, alpha=1)

        return F_1, F_2

class Cross_layer(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0
    ):
        super().__init__()
        self.d_model = hidden_dim
        # self.texture_enhance1 = Enhancement_texture(self.d_model,basic_conv1=Conv2d_Hori_Veri_Cross, basic_conv2=Conv2d_Diag_Cross, theta=0.8)
        # self.texture_enhance2 = Enhancement_texture(self.d_model, basic_conv1=Conv2d_Hori_Veri_Cross,
        #                                             basic_conv2=Conv2d_Diag_Cross, theta=0.8)
        self.texture_enhance1 = Enhancement_texture_LDC(self.d_model,self.d_model)
        self.texture_enhance2 = Enhancement_texture_LDC(self.d_model, self.d_model)
        self.Diff_enhance = Differential_enhance(self.d_model)
        self.Fuse = CrossMamba(hidden_dim)

    def forward(self,x1,x2):
        B, N, C = x1.shape
        x1 = x1.transpose(1, 2).view(B, C, 16, 16)
        x2 = x2.transpose(1, 2).view(B, C, 16, 16)
        Fuse = torch.add(x1, x2, alpha=1)

        TX_x1 = self.texture_enhance1(x1)
        TX_x2 = self.texture_enhance2(x2)

        DF_x1, DF_x2 = self.Diff_enhance(Fuse, x1,x2)
        F_1 = TX_x1 +DF_x1
        F_2 = TX_x2 +DF_x2

        F_1 = F_1.view(B,C, -1).transpose(1, 2)
        F_2 = F_2.view(B, C, -1).transpose(1, 2)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        F_1 = F_1.to(device)
        F_2 = F_2.to(device)
        F_1 = self.Fuse(F_1, 0, F_2)

        return F_1

# class SpatiotemporalAttentionFull(nn.Module):
#     def __init__(self, in_channels, inter_channels=None, dimension=2, sub_sample=False):
#         super(SpatiotemporalAttentionFull, self).__init__()
#         assert dimension in [2, ]
#         self.dimension = dimension
#         self.sub_sample = sub_sample
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels
#
#         if self.inter_channels is None:
#             self.inter_channels = in_channels // 2
#             if self.inter_channels == 0:
#                 self.inter_channels = 1
#
#         self.g = nn.Sequential(
#             nn.BatchNorm2d(self.in_channels),
#             nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                       kernel_size=1, stride=1, padding=0)
#         )
#
#         self.W = nn.Sequential(
#             nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
#                       kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(self.in_channels)
#         )
#         self.theta = nn.Sequential(
#             nn.BatchNorm2d(self.in_channels),
#             nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                       kernel_size=1, stride=1, padding=0),
#         )
#         self.phi = nn.Sequential(
#             nn.BatchNorm2d(self.in_channels),
#             nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
#                       kernel_size=1, stride=1, padding=0),
#         )
#         self.energy_time_1_sf = nn.Softmax(dim=-1)
#         self.energy_time_2_sf = nn.Softmax(dim=-1)
#         self.energy_space_2s_sf = nn.Softmax(dim=-2)
#         self.energy_space_1s_sf = nn.Softmax(dim=-2)
#
#     def forward(self, x1, x2):
#
#         batch_size = x1.size(0)
#         g_x11 = self.g(x1).reshape(batch_size, self.inter_channels, -1)
#         g_x12 = g_x11.permute(0, 2, 1)
#         g_x21 = self.g(x2).reshape(batch_size, self.inter_channels, -1)
#         g_x22 = g_x21.permute(0, 2, 1)
#
#         theta_x1 = self.theta(x1).reshape(batch_size, self.inter_channels, -1)
#         theta_x2 = theta_x1.permute(0, 2, 1)
#
#         phi_x1 = self.phi(x2).reshape(batch_size, self.inter_channels, -1)
#         phi_x2 = phi_x1.permute(0, 2, 1)
#
#         energy_time_1 = torch.matmul(theta_x1, phi_x2)
#         energy_time_2 = energy_time_1.permute(0, 2, 1)
#         energy_space_1 = torch.matmul(theta_x2, phi_x1)
#         energy_space_2 = energy_space_1.permute(0, 2, 1)
#
#         energy_time_1s = self.energy_time_1_sf(energy_time_1)
#         energy_time_2s = self.energy_time_2_sf(energy_time_2)
#         energy_space_2s = self.energy_space_2s_sf(energy_space_1)
#         energy_space_1s = self.energy_space_1s_sf(energy_space_2)
#
#         # energy_time_2s*g_x11*energy_space_2s = C2*S(C1) × C1*H1W1 × S(H1W1)*H2W2 = (C2*H2W2)' is rebuild C1*H1W1
#         y1 = torch.matmul(torch.matmul(energy_time_2s, g_x11), energy_space_2s).contiguous() # C2*H2W2
#         # energy_time_1s*g_x12*energy_space_1s = C1*S(C2) × C2*H2W2 × S(H2W2)*H1W1 = (C1*H1W1)' is rebuild C2*H2W2
#         y2 = torch.matmul(torch.matmul(energy_time_1s, g_x21), energy_space_1s).contiguous()
#         y1 = y1.reshape(batch_size, self.inter_channels, *x2.size()[2:])
#         y2 = y2.reshape(batch_size, self.inter_channels, *x1.size()[2:])
#         return x1 + self.W(y1), x2 + self.W(y2)
#
# class AgentAttention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
#                  agent_num=49, window=14, **kwargs):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = head_dim ** -0.5
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.agent_num = agent_num
#         self.window = window
#
#         self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
#                              padding=1, groups=dim)
#         self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
#         self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1))
#         self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window))
#         self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, agent_num))
#         self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, agent_num))
#         self.ac_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
#         self.ca_bias = nn.Parameter(torch.zeros(1, num_heads, 1, agent_num))
#         trunc_normal_(self.an_bias, std=.02)
#         trunc_normal_(self.na_bias, std=.02)
#         trunc_normal_(self.ah_bias, std=.02)
#         trunc_normal_(self.aw_bias, std=.02)
#         trunc_normal_(self.ha_bias, std=.02)
#         trunc_normal_(self.wa_bias, std=.02)
#         trunc_normal_(self.ac_bias, std=.02)
#         trunc_normal_(self.ca_bias, std=.02)
#         pool_size = int(agent_num ** 0.5)
#         self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
#
#     def forward(self, x):
#         """
#         Args:
#             x: input features with shape of (num_windows*B, N, C)
#             mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
#         """
#         x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))
#         b, n, c = x.shape
#         h = int(n ** 0.5)
#         w = int(n ** 0.5)
#         num_heads = self.num_heads
#         head_dim = c // num_heads
#         qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
#         q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
#         # q, k, v: b, n, c
#
#         agent_tokens = self.pool(q[:, 0:, :].reshape(b, h, w, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
#         q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
#         agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)
#
#         position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window), mode='bilinear')
#         position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
#         position_bias = position_bias1 + position_bias2
#         position_bias = torch.cat([self.ac_bias.repeat(b, 1, 1, 1), position_bias], dim=-1)
#         agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1))
#         agent_attn = self.attn_drop(agent_attn)
#         agent_v = agent_attn @ v
#
#         agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window), mode='bilinear')
#         agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
#         agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
#         agent_bias = agent_bias1 + agent_bias2
#         agent_bias = torch.cat([self.ca_bias.repeat(b, 1, 1, 1), agent_bias], dim=-2)
#         q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) )
#         q_attn = self.attn_drop(q_attn)
#         x = q_attn @ agent_v
#
#         x = x.transpose(1, 2).reshape(b, n, c)
#         v_ = v[:, :, 0:, :].transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
#         x[:, 0:, :] = x[:, 0:, :] + self.dwc(v_).permute(0, 2, 3, 1).reshape(b, n, c)
#
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x







if __name__ == "__main__":

    in_tensor1 = torch.ones((8, 256, 384))
    in_tensor2 = torch.ones((8, 256, 384,))
    Fuse = torch.add(in_tensor1, in_tensor2, alpha=1)

    Cross_layer = Cross_layer(hidden_dim=384)
    F_1 = Cross_layer(in_tensor1, in_tensor2)


    # sp_full = SpatiotemporalAttentionFull(in_channels=384)
    # output_full_x1, output_full_x2 = sp_full(F_1, F_2)
    #
    # agent = AgentAttention(dim=384)
    #
    # out1_agent = agent(F_1)

    print('out_tensor1:', F_1.shape)  # torch.Size([1, 2, 256, 31, 31])
    # print('out_tensor2:', F_2.shape)  # torch.Size([1, 2, 256, 7, 7])
    # print('output_full_x1:', output_full_x1.shape)  # torch.Size([1, 2, 256, 31, 31])
    # print('output_full_x1:', output_full_x2.shape)  # torch.Size([1, 2, 256, 7, 7])
    # print('out1_agent:', out1_agent.shape)  # torch.Size([1, 2, 256, 7, 7])