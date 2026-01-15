import torch
import torch.nn as nn
from timm.models.layers import DropPath, Mlp
import math
import os
import clip
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
class Attention_TI(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qkv_mem = None

    def forward(self, x, image_tokens, text_tokens):
        """
        x is a concatenated vector of template and search region features.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        q_mt, q_s = torch.split(q, [text_tokens, image_tokens], dim=2)
        k_mt, k_s = torch.split(k, [text_tokens, image_tokens], dim=2)
        v_mt, v_s = torch.split(v, [text_tokens, image_tokens], dim=2)

        # asymmetric mixed attention
        attn = (q_mt @ k_mt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_mt = (attn @ v_mt).transpose(1, 2).reshape(B, text_tokens, C)

        attn = (q_s @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_s = (attn @ v).transpose(1, 2).reshape(B, image_tokens, C)

        x = torch.cat([x_mt, x_s], dim=1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

# 应用于mdot的候选消除
def candidate_elimination_mdot(attn: torch.Tensor, tokens: torch.Tensor, lens_t: int, keep_ratio: float, global_index: torch.Tensor, box_mask_z: torch.Tensor):
    """
    Eliminate potential background candidates for computation reduction and noise cancellation.
    Args:
        attn (torch.Tensor): [B, num_heads, L_t + L_s, L_t + L_s], attention weights
        tokens (torch.Tensor):  [B, L_t + L_s, C], template and search region tokens
        lens_t (int): length of template
        keep_ratio (float): keep ratio of search region tokens (candidates)
        global_index (torch.Tensor): global index of search region tokens
        box_mask_z (torch.Tensor): template mask used to accumulate attention weights

    Returns:
        tokens_new (torch.Tensor): tokens after candidate elimination
        keep_index (torch.Tensor): indices of kept search region tokens
        removed_index (torch.Tensor): indices of removed search region tokens
    """
    lens_t_all = 1*lens_t
    lens_s = attn.shape[-1] - lens_t_all     # 减去两个模板长度
    bs, hn, _, _ = attn.shape

    lens_keep = math.ceil(keep_ratio * lens_s)
    if lens_keep == lens_s:
        return tokens, global_index, None

    attn_t = attn[:, :, :lens_t, lens_t_all:]   # 第一个模板和search的attention
    # attn_t2 = attn[:, :, lens_t:lens_t_all, lens_t_all:]   # 第二个模板

    if box_mask_z is not None:
        box_mask_z1 = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        # attn_t = attn_t[:, :, box_mask_z, :]
        attn_t = attn_t[box_mask_z1]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
        #
        # box_mask_z2 = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t2.shape[1], -1, attn_t2.shape[-1])
        # # attn_t = attn_t[:, :, box_mask_z, :]
        # attn_t2 = attn_t2[box_mask_z2]
        # attn_t2 = attn_t2.view(bs, hn, -1, lens_s)
        # attn_t2 = attn_t2.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s

        # attn_t = [attn_t[i, :, box_mask_z[i, :], :] for i in range(attn_t.size(0))]
        # attn_t = [attn_t[i].mean(dim=1).mean(dim=0) for i in range(len(attn_t))]
        # attn_t = torch.stack(attn_t, dim=0)
    else:
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # B, H, L-T, L_s --> B, L_s
        # attn_t2 = attn_t2.mean(dim=2).mean(dim=1)

    # attn_t = (attn_t + attn_t2)/2                # attention求均值
    attn_t = attn_t  # attention求均值
    # use sort instead of topk, due to the speed issue
    # https://github.com/pytorch/pytorch/issues/22812
    sorted_attn, indices = torch.sort(attn_t, dim=1, descending=True)               # 将attention map排序

    topk_attn, topk_idx = sorted_attn[:, :lens_keep], indices[:, :lens_keep]              # 前lens_keep个
    non_topk_attn, non_topk_idx = sorted_attn[:, lens_keep:], indices[:, lens_keep:]

    keep_index = global_index.gather(dim=1, index=topk_idx)
    removed_index = global_index.gather(dim=1, index=non_topk_idx)

    # separate template and search tokens
    tokens_t = tokens[:, :lens_t_all]           # 两个模板
    tokens_s = tokens[:, lens_t_all:]

    # obtain the attentive and inattentive tokens
    B, L, C = tokens_s.shape
    # topk_idx_ = topk_idx.unsqueeze(-1).expand(B, lens_keep, C)
    attentive_tokens = tokens_s.gather(dim=1, index=topk_idx.unsqueeze(-1).expand(B, -1, C))
    # inattentive_tokens = tokens_s.gather(dim=1, index=non_topk_idx.unsqueeze(-1).expand(B, -1, C))

    # compute the weighted combination of inattentive tokens
    # fused_token = non_topk_attn @ inattentive_tokens

    # concatenate these tokens
    # tokens_new = torch.cat([tokens_t, attentive_tokens, fused_token], dim=0)
    tokens_new = torch.cat([tokens_t, attentive_tokens], dim=1)

    return tokens_new, keep_index, removed_index


class Block_TI(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_TI(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, image_tokens, text_tokens):
        B, N, C = x.shape
        ce_template_mask = None
        global_index_s = torch.linspace(0, image_tokens - 1, image_tokens).to(x.device)
        global_index_s = global_index_s.repeat(B, 1)
        x, attn = self.attn(self.norm1(x), image_tokens, text_tokens)
        x = x + self.drop_path1(x)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        x, global_index_search, removed_index_search = candidate_elimination_mdot(attn, x, 1, 0.8, global_index_s, ce_template_mask)


        return x, global_index_search


if __name__ == "__main__":

    in_tensor1 = torch.ones((8, 64, 384))
    in_tensor2 = torch.ones((8, 1, 384))
    in_tensor = torch.cat((in_tensor2, in_tensor1), dim=1)
    Block_TI = Block_TI(dim=384, num_heads=8)
    x, _ = Block_TI(in_tensor, 64, 1)
    # from PIL import Image
    # model,  preprocess = clip.load("ViT-L/14@336px", device='cpu')
    # image = preprocess(Image.open("00000001.jpg")).unsqueeze(0).to("cpu")
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to("cpu")
    #
    # with torch.no_grad():
    #     image_features = model.encode_image(image)
    #     text_features = model.encode_text(text)
    #
    # in_tensor = torch.cat((text_features, image_features), dim=1)
    # Block_TI = Block_TI(dim=384, num_heads=8)
    # x, _ = Block_TI(in_tensor, 64, 1)




    #
    # print('out_tensor1:', x.shape)  # torch.Size([1, 2, 256, 31, 31])
    # print('out_tensor2:', x.shape)  # torch.Size([1, 2, 256, 7, 7])