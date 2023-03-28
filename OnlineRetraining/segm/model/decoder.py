import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from timm.models.layers import trunc_normal_

from segm.model.blocks import Block, FeedForward
from segm.model.utils import init_weights


class DecoderLinear(nn.Module):
    def __init__(self, n_cls, patch_size, d_encoder):
        super().__init__()

        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_cls = n_cls

        self.head = nn.Linear(self.d_encoder, n_cls)
        self.apply(init_weights)

    @torch.jit.ignore
    def no_weight_decay(self):
        return set()

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size
        x = self.head(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=GS)

        return x


class MaskTransformer(nn.Module):
    def __init__(
            self,
            n_cls,
            patch_size,
            d_encoder,
            n_layers,
            n_heads,
            d_model,
            d_ff,
            drop_path_rate,
            dropout,
    ):
        super().__init__()
        self.d_encoder = d_encoder
        self.patch_size = patch_size
        self.n_layers = n_layers
        self.n_cls = n_cls
        self.d_model = d_model
        self.d_ff = d_ff
        self.scale = d_model ** -0.5

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout, dpr[i]) for i in range(n_layers)]
        )

        self.cls_emb = nn.Parameter(torch.randn(1, n_cls, d_model))
        self.proj_dec = nn.Linear(d_encoder, d_model)

        self.proj_patch = nn.Parameter(self.scale * torch.randn(d_model, d_model))
        self.proj_classes = nn.Parameter(self.scale * torch.randn(d_model, d_model))

        self.decoder_norm = nn.LayerNorm(d_model)
        self.mask_norm = nn.LayerNorm(n_cls)

        self.apply(init_weights)
        trunc_normal_(self.cls_emb, std=0.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_emb"}

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks

    def get_attention_map(self, x, layer_id):
        if layer_id >= self.n_layers or layer_id < 0:
            raise ValueError(
                f"Provided layer_id: {layer_id} is not valid. 0 <= {layer_id} < {self.n_layers}."
            )
        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for i, blk in enumerate(self.blocks):
            if i < layer_id:
                x = blk(x)
            else:
                return blk(x, return_attention=True)


class GradientClipping(nn.Module):
    def __init__(self, start_value, patch_size):
        super().__init__()
        self.start_value = start_value
        self.patch_size = patch_size

    def forward(self, seg_pred, seg_gt, criterion):
        ori_loss = criterion(seg_pred, seg_gt)
        detach_loss = ori_loss.detach().clone()

        mean_loss = detach_loss.mean()

        # set start loss clamp threshold
        if mean_loss > self.start_value:
            return ori_loss, ori_loss

        b, h, w = detach_loss.shape

        # all batch average
        detach_loss = detach_loss.mean(dim=0).unsqueeze(0)
        local_mean = F.avg_pool2d(detach_loss.unsqueeze(1), kernel_size=self.patch_size,
                                  stride=self.patch_size, padding=h % self.patch_size,
                                  count_include_pad=False).squeeze(1)
        local_mean = torch.maximum(local_mean, mean_loss)
        local_mean = torch.repeat_interleave(local_mean, b, dim=0)
        local_mean = torch.repeat_interleave(local_mean, self.patch_size, dim=1)
        local_mean = torch.repeat_interleave(local_mean, self.patch_size, dim=2)

        clamp_loss = ori_loss - local_mean
        clamp_loss = torch.clamp(clamp_loss, None, 0)
        loss = clamp_loss + local_mean

        return ori_loss, loss


class MultiMaskTransformer(MaskTransformer):
    def __init__(self,
                 n_cls,
                 patch_size,
                 d_encoder,
                 n_layers,
                 n_heads,
                 d_model,
                 d_ff,
                 drop_path_rate,
                 dropout):
        super(MultiMaskTransformer, self).__init__(n_cls,
                                                   patch_size,
                                                   d_encoder,
                                                   n_layers,
                                                   n_heads,
                                                   d_model,
                                                   d_ff,
                                                   drop_path_rate,
                                                   dropout, )

    def forward(self, x, im_size):
        H, W = im_size
        GS = H // self.patch_size

        x = self.proj_dec(x)
        cls_emb = self.cls_emb.expand(x.size(0), -1, -1)
        x = torch.cat((x, cls_emb), 1)
        for blk in self.blocks[:-1]:
            x = blk(x)
        x2 = x
        for blk in self.blocks[-1:]:
            x2 = blk(x2)

        masks1 = self.cls_forward(x, GS)
        masks2 = self.cls_forward(x2, GS)

        return masks1, masks2

    def cls_forward(self, x, GS):
        x = self.decoder_norm(x)

        patches, cls_seg_feat = x[:, : -self.n_cls], x[:, -self.n_cls:]
        patches = patches @ self.proj_patch
        cls_seg_feat = cls_seg_feat @ self.proj_classes

        patches = patches / patches.norm(dim=-1, keepdim=True)
        cls_seg_feat = cls_seg_feat / cls_seg_feat.norm(dim=-1, keepdim=True)

        masks = patches @ cls_seg_feat.transpose(1, 2)
        masks = self.mask_norm(masks)
        masks = rearrange(masks, "b (h w) n -> b n h w", h=int(GS))

        return masks
