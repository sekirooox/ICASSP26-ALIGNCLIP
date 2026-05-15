from ast import Gt
import numpy as np
from mmcv.cnn import ConvModule
from mmseg.ops import Upsample, resize

from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from typing import Optional
import math
from functools import partial
from mmcv.runner import auto_fp16, force_fp32
import matplotlib.pyplot as plt

from timm.models.layers import trunc_normal_
import matplotlib.pyplot as plt
from mmseg.models.losses import accuracy

from models.decode_heads.utils import positional_encoding

def gaussian(x, mu, sigma):
    return torch.exp(-0.5 * (x / sigma)**2)

def trunc_normal_init(module: nn.Module,
                      mean: float = 0,
                      std: float = 1,
                      a: float = -2,
                      b: float = 2,
                      bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        trunc_normal_(module.weight, mean, std, a, b)  # type: ignore
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)  # type: ignore

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class TPN_Decoder(TransformerDecoder):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None):
        output = tgt
        attns = []
        outputs = []
        for mod in self.layers:
            output, attn = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            attns.append(attn)
            outputs.append(output)
        if self.norm is not None: # not do
            output = self.norm(output)

        return outputs, attns

class TPN_DecoderLayer(TransformerDecoderLayer):
    def __init__(self, **kwargs):
        super(TPN_DecoderLayer, self).__init__(**kwargs)
        del self.multihead_attn
        self.multihead_attn = Attention(
            kwargs['d_model'], num_heads=kwargs['nhead'], qkv_bias=True, attn_drop=0.1)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        tgt2, attn2 = self.multihead_attn(
            tgt.transpose(0, 1), memory.transpose(0, 1), memory.transpose(0, 1))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn2

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, xq, xk, xv):
        B, Nq, C = xq.size() 
        Nk = xk.size()[1]
        Nv = xv.size()[1]
        q = self.q(xq).reshape(B, Nq, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(xk).reshape(B, Nk, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(xv).reshape(B, Nv, self.num_heads,
                                      C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_save = attn.clone()
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose(0, 1), attn_save.sum(dim=1) / self.num_heads
        
class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@HEADS.register_module()
class ATMSingleHeadSeg(BaseDecodeHead):
    def __init__(
            self,
            img_size,
            in_channels,
            seen_idx,
            all_idx,
            embed_dims=768,
            num_layers=3,
            num_heads=8,
            use_stages=1,
            **kwargs,
    ):
        super(ATMSingleHeadSeg, self).__init__(
            in_channels=in_channels, **kwargs)
                

        def create_decoder():
            """Build a transformer decoder stack for one cascade branch."""
            decoder_layer = TPN_DecoderLayer(d_model=dim, nhead=nhead, dim_feedforward=dim * 4)
            return TPN_Decoder(decoder_layer, num_layers)

        self.image_size = img_size
        self.seen_idx = seen_idx
        self.all_idx = all_idx
        self.unseen_idx = [idx for idx in all_idx if idx not in seen_idx]
        self.use_stages=use_stages

        nhead = num_heads
        dim = embed_dims

        # One decoder per cascade branch.
        self.cascade_decoders = nn.ModuleList([create_decoder() for _ in range(3)])

        if hasattr(self, 'conv_seg'):
            delattr(self, 'conv_seg')

        # Branch-specific query and lateral projections.
        self.q_proj_g =  nn.Linear(2 * dim,dim)
        self.q_proj_r1 = nn.Linear(3 * dim,dim)
        self.q_proj_r2 = nn.Linear(3 * dim,dim)

        self.lateral_g_proj = nn.Linear(dim * 2, dim)
        self.lateral_r1_proj = nn.Linear(dim * 3, dim)
        self.lateral_r2_proj = nn.Linear(dim * 3, dim)

    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m, std=.02, bias=0)
            elif isinstance(m, nn.LayerNorm):
                constant_init(m, val=1.0, bias=0.0)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg, self_training=False, st_mask=None):
        seg_logits = self.forward(inputs)

        if self_training:
            pseudo_semantic_masks = seg_logits['pred_masks'].clone().detach().sigmoid()
            pseudo_semantic_masks[:, self.seen_idx, :, :] = -1
            pseudo_semantic_seg = pseudo_semantic_masks.argmax(dim=1).unsqueeze(1)
            # generate pseudo labels for "transductive" setting
            gt_semantic_seg[gt_semantic_seg==-1] = pseudo_semantic_seg[gt_semantic_seg==-1]
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)
        else:
            gt_semantic_seg[gt_semantic_seg==-1] = 255
            losses = self.losses(seg_logits, gt_semantic_seg)

        return losses

    def forward_test(self, inputs, img_metas, test_cfg, self_training):
        return self.forward(inputs, self_training)

    def forward(self, inputs_both, self_training=None):

        inputs = inputs_both[0][0]
        test=inputs_both[0][0]
        B,D,H,W=inputs_both[0][0][0].shape
        cls_token = inputs_both[0][1]
        text_token = inputs_both[1]
        rlb_token_1= inputs_both[0][2].reshape(B,D,-1).permute(0,2,1)
        rlb_token_2= inputs_both[0][3].reshape(B,D,-1).permute(0,2,1)

        laterals = [self.d4_to_d3(stage) if stage.dim() > 3 else stage for stage in inputs[:self.use_stages]]

        lateral_g,lateral_r1,lateral_r2=self.get_lateral(laterals[-1],cls_token,rlb_token_1,rlb_token_2)

        lateral_g=self.lateral_g_proj(lateral_g)
        lateral_r1=self.lateral_r1_proj(lateral_r1)
        lateral_r2=self.lateral_r2_proj(lateral_r2)

        # Build one query stream for each cascade branch.
        q_g, q_r1, q_r2=self.get_qs(text_token, cls_token,rlb_token_1,rlb_token_2)
        q_g=self.q_proj_g(q_g).permute(1,0,2)
        q_r1=self.q_proj_r1(q_r1).permute(1,0,2)
        q_r2=self.q_proj_r2(q_r2).permute(1,0,2)

          
        decoder_output_g = self.decode(q_g, self.cascade_decoders[0],lateral_g, [])
        decoder_output_r1 = self.decode(q_r1, self.cascade_decoders[1],lateral_r1, [])
        decoder_output_r2 = self.decode(q_r2, self.cascade_decoders[2],lateral_r2, [])

        decoder_output_g=F.interpolate(decoder_output_g[-1],size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)
        decoder_output_r1=F.interpolate(decoder_output_r1[-1],size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)
        decoder_output_r2=F.interpolate(decoder_output_r2[-1],size=(self.image_size,self.image_size),mode='bilinear',align_corners=False)

        stacked_attn=torch.stack(
            [decoder_output_g,decoder_output_r1,decoder_output_r2],dim=0
        )
        # Sum the branch outputs to form the final mask logits.
        summed_attn =stacked_attn.sum(dim=0)

        out = {}    
        out["pred_masks"]=summed_attn
 
        if not self.training:
            if self_training:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx)
            else:
                out["pred"] = self.semantic_inference(out["pred_masks"], self.seen_idx, 0.1)
            return out["pred"]
        return out
    
    def decode(self,q, decoder, lateral,decoder_output):
        """Convert decoder attention maps into branch mask predictions."""
        _, attn_ = decoder(q, lateral.transpose(0, 1))
        for attn in attn_:
            attn = attn.transpose(-1, -2)
            attn = self.d3_to_d4(attn)
            decoder_output.append(attn)
        return decoder_output
    
    def get_lateral(self,lateral,cls,rlb_token_1,rlb_token_2):
        """Fuse local features with global and region-level context."""
        B,L,D=lateral.shape
        H=W=int(math.sqrt(L))

        rlb_token_1=self.d4_to_d3(
            F.interpolate(
            self.d3_to_d4(rlb_token_1),
            size=(H,W),
            mode='bilinear',align_corners=False
            )
        )
        rlb_token_2=self.d4_to_d3(
            F.interpolate(
            self.d3_to_d4(rlb_token_2),
            size=(H,W),
            mode='bilinear',align_corners=False
            )
        )
        
        lateral_g=torch.cat([
            lateral,
            cls.unsqueeze(1).expand(-1,L,-1),
        ],dim=-1)

        lateral_r1=torch.cat([
            lateral,
            cls.unsqueeze(1).expand(-1,L,-1),
            rlb_token_1,
        ],dim=-1)

        lateral_r2=torch.cat([
            lateral,
            cls.unsqueeze(1).expand(-1,L,-1),
            rlb_token_2,
        ],dim=-1)
        return lateral_g,lateral_r1,lateral_r2
    
    def get_qs(self,q,cls,rlb_token_1,rlb_token_2):
        """Construct branch queries from text, global, and region features."""
        C, dim = q.shape
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)

        q1 = torch.einsum("bd,bcd->bcd", cls, q)

        q2 = torch.einsum("bkd,bcd->bkcd",rlb_token_1,q).mean(dim=1)

        q3 = torch.einsum("bkd,bcd->bkcd",rlb_token_2,q).mean(dim=1)

        q_g=torch.cat([
            q,q1
        ],dim=-1)

        q_r1=torch.cat([
            q,q1,q2,
        ],dim=-1)

        q_r2=torch.cat([
            q,q1,q3
        ],dim=-1)
        
        return q_g,q_r1,q_r2

    def semantic_inference(self, mask_pred, seen_idx, weight=0.0):
        mask_pred = mask_pred.sigmoid()
        mask_pred[:,seen_idx] = mask_pred[:,seen_idx] - weight
        return mask_pred

    @torch.jit.unused
    def _set_aux_loss(self, outputs_seg_masks):
        return [
            {"pred_masks": a}
            # for a in zip(outputs_seg_masks[:-1])
            for a in outputs_seg_masks[:-1]
        ]

    def d3_to_d4(self, t):
        n, hw, c = t.size()
        if hw % 2 != 0:
            t = t[:, 1:]
        h = w = int(math.sqrt(hw))
        return t.transpose(1, 2).reshape(n, c, h, w)

    def d4_to_d3(self, t):
        return t.flatten(-2).transpose(-1, -2)

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label, num_classes=None):
        """Compute segmentation loss."""
        if isinstance(seg_logit, dict):
            # atm loss
            seg_label = seg_label.squeeze(1)

            loss = self.loss_decode(
                seg_logit,
                seg_label,
                ignore_index = self.ignore_index)

            loss['acc_seg'] = accuracy(seg_logit["pred_masks"], seg_label, ignore_index=self.ignore_index)
            return loss
