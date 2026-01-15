"""
Basic mamda_fetrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.ostrack.vit_ce_mdot import vit_large_patch16_224_ce_mdot, vit_base_patch16_224_ce_mdot
from lib.models.ostrack.vit_ce_mdot2s import vit_large_patch16_224_ce_mdot2s, vit_base_patch16_224_ce_mdot2s
from lib.models.ostrack.vit_ce_threemdot import vit_base_patch16_224_ce_mdot_three
from lib.models.layers.head import build_box_head
from lib.utils.box_ops import box_xyxy_to_cxcywh
# LXQ 测试三机修改
# from lib.models.mamba_fetrack.models_mamba_ori import create_block
# from lib.models.mamba_fetrack.models_mamba_three import create_block
from timm.models import create_model
import torch.nn.functional as F
from lib.models.mamba_fetrack.mamba_cross import CrossMamba
from thop import profile
from lib.models.mamba_fetrack.Mamba_VIT import Block_VIT
from lib.models.layers.category_embedding import Category_embedding


class Mamba_FEtrack_three(nn.Module):
    """ This is the base class for mamda_fetrack """

    def __init__(self, visionmamba, cross_mamba, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = visionmamba
        self.block_VIT = Block_VIT(dim=384, num_heads=8)
        self.cross_mamba = cross_mamba
        self.box_head = box_head
        self.block_VIT = Block_VIT(dim=384, num_heads=8)
        self.aux_loss = aux_loss
        self.head_type = head_type
        self.category_embedding = Category_embedding()
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                template2: torch.Tensor,  # 双机时有
                template3: torch.Tensor,  # 三机时有
                search: torch.Tensor,
                # search2: torch.Tensor,             # 双搜索区域
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        #############################LXQ 新增 ##############################

        # class_desbb_z, color_desbb_z, material_desbb_z, texture_desbb_z, attention_desbb_z = self.category_embedding(template)
        # text_code = torch.cat([class_desbb_z, color_desbb_z, material_desbb_z, texture_desbb_z], dim=1)
        # class_des, color_des, material_des, texture_des, attention_des = self.category_embedding(template_clip, search_clip)

        rgb_feature = self.backbone.forward_features(z=template,
                                    z2=template2,  # 双机时有
                                    z3=template3,
                                    x=search,
                                    # x2=search2,             # 双搜索区域
                                    inference_params=None, if_random_cls_token_position=False,
                                    if_random_token_rank=False, mask=None)




        # rgb_feature = self.block_VIT(rgb_feature)

        ############注释 ####################
        # event_feature = self.backbone.forward_features(z=event_template, x=event_search, z2=template, # [B, 320, 384]
        #                                                inference_params=None, if_random_cls_token_position=False,
        #                                                if_random_token_rank=False, mask=None)

        # rgb_feature = self.backbone.forward_features( z=template, x=search,                                                                     #[B, 320, 384]
        #                                         inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)
        # event_feature = self.backbone.forward_features(z=event_template, x=event_search,                                                        #[B, 320, 384]
        #                                         inference_params=None, if_random_cls_token_position=False, if_random_token_rank=False)

        residual_event_f = 0
        residual_rgb_f = 0
        # event_f = self.cross_mamba(event_feature,residual_event_f,rgb_feature) + event_feature
        # rgb_f = self.cross_mamba(rgb_feature,residual_rgb_f,event_feature) + rgb_feature

        # event_searh = event_f[:, -self.feat_len_s:]
        rgb_search = rgb_feature[:, -self.feat_len_s:]
        ###  LXQ  原来 ##############
        # x = torch.cat((event_searh,rgb_search),dim=-1)
        ###  LXQ  修改  ##############
        x = rgb_search
        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        search_feature = cat_feature
        opt = (search_feature.unsqueeze(-1)).permute(
            (0, 3, 2, 1)).contiguous()  # opt.shape = torch.Size([B, 1, 384, 256])
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)  # opt_feat.shape = torch.Size([B, 384, 16, 16])

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head

            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_mamba_fetrack_three(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('Mamba_FETrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    # if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
    #     backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
    #     backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                        ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                        ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                        )
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
    #     backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                         ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                         ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                         )
    #
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_mdot':  # 创建mdot的模型
    #     backbone = vit_base_patch16_224_ce_mdot(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                             ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                             ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                             )
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce_mdot':
    #     backbone = vit_large_patch16_224_ce_mdot(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                              ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                              ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                              )
    #
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_mdot2s':  # 创建mdot的模型
    #     backbone = vit_base_patch16_224_ce_mdot2s(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                               ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                               ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                               )
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce_mdot2s':
    #     backbone = vit_large_patch16_224_ce_mdot2s(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                                ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                                ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                                )
    #
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_mdot_three':  # 三机
    #     backbone = vit_base_patch16_224_ce_mdot_three(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
    #                                                   ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
    #                                                   ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
    #                                                   )
    #
    #     hidden_dim = backbone.embed_dim
    #     patch_start_index = 1
    #
    # else:
    #     raise NotImplementedError

    backbone = create_model(model_name=cfg.MODEL.BACKBONE.TYPE, pretrained=pretrained, num_classes=1000,
                            drop_rate=0.0, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, drop_block_rate=None, img_size=256
                            )
    hidden_dim = 384
    cross_mamba = CrossMamba(hidden_dim)
    ##########  LXQ  修改  build_box_head(cfg, hidden_dim*2) ######################
    box_head = build_box_head(cfg, hidden_dim)
    model = Mamba_FEtrack_three(
        backbone,
        cross_mamba,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'Mamba_FETrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model


