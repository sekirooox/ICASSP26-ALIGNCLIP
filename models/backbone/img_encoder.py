import math
import torch
import torch.nn.functional as F
from torch.nn import Dropout
from torch import nn
from mmseg.models.builder import BACKBONES
from functools import reduce
from operator import mul

from .utils import LayerNorm, Transformer
@BACKBONES.register_module()

class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution=224, patch_size=32, width=768, layers=12, heads=12, output_dim=512, drop_path_rate=0.0, out_indices=[3, 5, 7, 11], pretrained=None, get_embeddings=False, **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings

        self.transformer = Transformer(width, layers, heads, drop_path_rate=drop_path_rate)

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        embed_dim = width
        self.patch_size = patch_size

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(pretrained, map_location='cpu').float().state_dict()

            state_dict = {} #new model

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict['positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)   upsample the positional_embedding for larger input
                    print(f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}')
                    cls_pos = state_dict["positional_embedding"][0:1, :]
                    if self.patch_size == 16:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 14, 14, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    elif self.patch_size == 32:
                        spatial_pos = F.interpolate(state_dict["positional_embedding"][1:,].reshape(1, 7, 7, 768).permute(0, 3, 1, 2), size=(self.spatial_size, self.spatial_size), mode='bilinear')
                    else:
                        assert AttributeError('Patch Size should be 16 or 32')
                    spatial_pos = spatial_pos.reshape(768, self.spatial_size*self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos], dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict['positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer') # it should be nothing is misaligned


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0,:] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[1:,].reshape(1, self.spatial_size, self.spatial_size, C).permute(0, 3, 1, 2), size=(H, W), mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H*W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        features = []
        outs = []
        for i, blk in enumerate(self.transformer.resblocks):
            x = blk(x)
            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(dim=1, keepdim=True)
                features.append(visual_embedding)

            outs.append(tuple(features))

            global_embedding = global_embedding / global_embedding.norm(dim=1, keepdim=True)
            outs.append(global_embedding) 

        return outs

@BACKBONES.register_module()
class VPTCLIPVisionTransformer(nn.Module):

    def __init__(self,
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=12,
                output_dim=512,
                drop_path_rate=0.0,
                out_indices=[3, 5, 7, 11],
                pretrained=None,
                get_embeddings=False,
                num_tokens=20,
                prompt_dim=512,
                total_d_layer=11,
                region_level_bridge_sizes=[16,64],# 2x2 4x4
                 **kwargs):
        super().__init__()
        self.pretrained = pretrained
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=width,
                               kernel_size=patch_size,
                               stride=patch_size,
                               bias=False)

        scale = width**-0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(
            (input_resolution // patch_size)**2 + 1, width))
        self.spatial_size = input_resolution // patch_size
        self.ln_pre = LayerNorm(width)
        self.get_embeddings = get_embeddings
        self.num_layers = layers

        # Visual prompt tuning settings.
        self.num_tokens = num_tokens
        self.prompt_dim = prompt_dim
        self.total_d_layer = total_d_layer

        self.region_level_bridge_sizes = region_level_bridge_sizes

        self.region_level_bridge_hws = [int(
            math.sqrt(region_level_bridge_size))
            for region_level_bridge_size in self.region_level_bridge_sizes
            ]
        
        # disable the gradient update
        self.region_level_bridges = [
            nn.Parameter(torch.zeros(
            region_level_bridge_size, prompt_dim),requires_grad=True)
            for region_level_bridge_size in self.region_level_bridge_sizes
        ]

        # Restrict each bridge token to its assigned spatial region.
        visual_mask = self.gen_attention_mask(
            num_tokens=self.num_tokens,
            spatial_size=self.spatial_size,
            region_level_bridge_sizes=self.region_level_bridge_sizes,
            region_level_bridge_hws=self.region_level_bridge_hws,
        )
        
        self.transformer = Transformer(
            width,
            layers,
            heads,
            drop_path_rate=drop_path_rate,
            attn_mask=visual_mask,
        )

        self.out_indices = out_indices

        if get_embeddings:
            self.ln_post = LayerNorm(width)
            self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # Add the prompt parameters
        self._init_prompt(patch_size, self.num_tokens, self.prompt_dim,
                          self.total_d_layer)
    
    def gen_attention_mask(
        self,
        num_tokens: int,
        spatial_size: int,
        region_level_bridge_sizes: list,
        region_level_bridge_hws: list
    ) -> torch.Tensor:
        """Build an attention mask for region-level bridge tokens."""
        assert len(region_level_bridge_sizes) == len(region_level_bridge_hws)
        assert all(hw > 0 for hw in region_level_bridge_hws)
        assert all(size > 0 for size in region_level_bridge_sizes)

        cls_token_size = 1
        patch_count = spatial_size ** 2
        rlb_total = sum(region_level_bridge_sizes)
        total_tokens = cls_token_size + num_tokens + patch_count + rlb_total

        visual_mask = torch.zeros((total_tokens, total_tokens), dtype=torch.float32)

        # Compute the token ranges for each bridge scale.
        rlb_starts = []
        current_start = total_tokens - rlb_total
        for size in region_level_bridge_sizes:
            rlb_starts.append(current_start)
            current_start += size

        visual_mask[:, rlb_starts[0]:] = float("-inf")
        visual_mask[rlb_starts[0]:, :] = float("-inf")

        # Allow each bridge token to attend only to its spatial patch group.
        patch_start = cls_token_size + num_tokens

        for scale_idx, (rlb_size, rlb_hw) in enumerate(zip(region_level_bridge_sizes, region_level_bridge_hws)):
            stride = spatial_size // rlb_hw
            rlb_start = rlb_starts[scale_idx]

            for i in range(rlb_hw):
                for j in range(rlb_hw):
                    rlb_pos = rlb_start + i * rlb_hw + j

                    tmp_mask = torch.zeros((spatial_size, spatial_size))
                    row_start, row_end = i * stride, (i + 1) * stride
                    col_start, col_end = j * stride, (j + 1) * stride
                    tmp_mask[row_start:row_end, col_start:col_end] = 1

                    patch_indices = []
                    for row in range(row_start, row_end):
                        for col in range(col_start, col_end):
                            patch_idx = patch_start + row * spatial_size + col
                            patch_indices.append(patch_idx)
                    patch_indices = torch.tensor(patch_indices, dtype=torch.long)

                    visual_mask[rlb_pos, patch_indices] = 0
                    visual_mask[patch_indices, rlb_pos] = 0

                    all_patches = torch.arange(patch_start, patch_start + patch_count)
                    other_patches = all_patches[~torch.isin(all_patches, patch_indices)]
                    visual_mask[rlb_pos, other_patches] = float("-inf")
                    visual_mask[other_patches, rlb_pos] = float("-inf")

            for pos in range(rlb_start, rlb_start + rlb_size):
                visual_mask[pos, pos] = 0

        # Prevent bridge tokens from different scales from mixing.
        for i in range(len(rlb_starts) - 1):
            current_end = rlb_starts[i] + region_level_bridge_sizes[i]
            next_start = rlb_starts[i + 1]
            visual_mask[current_end:next_start, next_start:] = float("-inf")
            visual_mask[next_start:, current_end:next_start] = float("-inf")
        
        return visual_mask

    def _init_prompt(self, patch, num_tokens, prompt_dim, total_d_layer):
        patch_size = []
        patch_size.append(patch)
        patch_size.append(patch)
        val = math.sqrt(
            6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

        if total_d_layer >= 0:
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, num_tokens, prompt_dim))
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if total_d_layer > 0:  # noqa
                self.deep_prompt_embeddings = nn.Parameter(
                    torch.zeros(total_d_layer, num_tokens, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

            self.prompt_proj = nn.Linear(prompt_dim, prompt_dim)
            nn.init.kaiming_normal_(self.prompt_proj.weight,
                                    a=0,
                                    mode='fan_out')
            self.prompt_norm = LayerNorm(prompt_dim, eps=1e-6)
            self.prompt_dropout = Dropout(0.1)

    def init_weights(self, pretrained=None):
        pretrained = pretrained or self.pretrained
        if isinstance(pretrained, str):
            checkpoint = torch.jit.load(
                pretrained, map_location='cpu').float().state_dict()

            state_dict = {}

            for k in checkpoint.keys():
                if k.startswith('visual.'):
                    new_k = k.replace('visual.', '')
                    state_dict[new_k] = checkpoint[k]

            if 'positional_embedding' in state_dict.keys():
                if self.positional_embedding.shape != state_dict[
                        'positional_embedding'].shape:
                    # (1025, 768)                      (197, 768)
                    print(
                        f'Resize the pos_embed shape from {state_dict["positional_embedding"].shape} to {self.positional_embedding.shape}'
                    )
                    cls_pos = state_dict["positional_embedding"][0:1, :]

                    spatial_pos = F.interpolate(
                        state_dict["positional_embedding"][
                            1:,
                        ].reshape(1, 14, 14, 768).permute(0, 3, 1, 2),
                        size=(self.spatial_size, self.spatial_size),
                        mode='bilinear')
                    spatial_pos = spatial_pos.reshape(
                        768,
                        self.spatial_size * self.spatial_size).permute(1, 0)
                    positional_embedding = torch.cat([cls_pos, spatial_pos],
                                                     dim=0)
                    state_dict['positional_embedding'] = positional_embedding
                    assert self.positional_embedding.shape == state_dict[
                        'positional_embedding'].shape

            u, w = self.load_state_dict(state_dict, False)
            print(u, w, 'are misaligned params in vision transformer')

            # Initialize bridge tokens from the CLIP class embedding.
            for i,region_level_bridge_size in enumerate(self.region_level_bridge_sizes):
                region_level_bridge_init = state_dict['class_embedding'].repeat(
                    region_level_bridge_size, 1)

                self.region_level_bridges[i].data = region_level_bridge_init + cls_pos.repeat(
                    region_level_bridge_size, 1)
            

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        B, C, H, W = x.shape
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)# [B,HW,D]
        x = torch.cat([
            self.class_embedding.to(x.dtype) + torch.zeros(
                x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), 
                x
        ],dim=1)# [B,1+HW,D]

        pos = self.positional_embedding.to(x.dtype)
        cls_pos = pos[0, :] + self.class_embedding.to(x.dtype)
        spatial_pos = F.interpolate(pos[
            1:,
        ].reshape(1, self.spatial_size, self.spatial_size,
                  C).permute(0, 3, 1, 2),
                                    size=(H, W),
                                    mode='bilinear')
        spatial_pos = spatial_pos.reshape(1, C, H * W).permute(0, 2, 1)
        pos = torch.cat([cls_pos.reshape(1, 1, C), spatial_pos], dim=1)
        x = x + pos

        # Append region-level bridge tokens after patch tokens.
        region_level_bridge = torch.cat(list(self.region_level_bridges), dim=0).to(x.device)
        x = torch.cat([x, region_level_bridge.expand(B, -1, -1)], dim=1)
        x = self.ln_pre(x)


        # Insert prompt tokens after the class token.
        if self.total_d_layer >= 0:
            x = torch.cat((x[:, :1, :],
                           self.prompt_dropout(
                            self.prompt_proj(self.prompt_embeddings).expand(
                                   B, -1, -1)), 
                            x[:, 1:, :]),
                          dim=1)

        x = x.permute(1, 0, 2)

        features = []
        outs = []
        if self.total_d_layer == 0:
            for i, blk in enumerate(self.transformer.resblocks):
                x = blk(x)
                if len(self.out_indices) > 1:
                    if i in self.out_indices:
                        xp = x.permute(1, 0,
                                       2)[:, 1 + self.num_tokens:, :].permute(
                                           0, 2, 1).reshape(B, -1, H, W)
                        features.append(xp.contiguous())
        elif self.total_d_layer > 0:
            x, features = self.forward_deep_prompt(x, features, H, W)

        if self.get_embeddings:
            x = x.permute(1, 0, 2)
            x = self.ln_post(x)
            x = x @ self.proj

            global_embedding = x[:, 0]
            visual_embedding = x[:, 1 + \
                                self.num_tokens:-sum(self.region_level_bridge_sizes)].reshape(
                                B, H, W, -1).permute(0, 3, 1, 2)
            region_level_bridges = x[:,-sum(self.region_level_bridge_sizes):]

            region_level_bridge_1= region_level_bridges[:,:self.region_level_bridge_sizes[0],:].reshape(
                B, self.region_level_bridge_hws[0],
                self.region_level_bridge_hws[0],-1).permute(0, 3, 1, 2)

            region_level_bridge_2= region_level_bridges[:,self.region_level_bridge_sizes[0]:,:].reshape(
                B, self.region_level_bridge_hws[1],
                self.region_level_bridge_hws[1],-1).permute(0, 3, 1, 2)

            if len(self.out_indices) == 1:
                visual_embedding = visual_embedding / visual_embedding.norm(
                    dim=1, keepdim=True)
                features.append(visual_embedding)

            outs.append(tuple(features))
            global_embedding = global_embedding / global_embedding.norm(
                dim=1, keepdim=True)
            outs.append(global_embedding)

            region_level_bridge_1 = region_level_bridge_1 / region_level_bridge_1.norm(
                dim=1, keepdim=True)
            outs.append(region_level_bridge_1)

            region_level_bridge_2 = region_level_bridge_2 / region_level_bridge_2.norm(
                dim=1, keepdim=True)
            outs.append(region_level_bridge_2)
        return outs

    def forward_deep_prompt(self,
                            embedding_output,
                            features,
                            H,
                            W,
                            out_last=False):
        """Apply deep prompt injection across transformer layers."""
        B = embedding_output.shape[1]

        for i in range(self.num_layers):
            if i == 0:
                hidden_states = self.transformer.resblocks[i](embedding_output)
            elif i <= self.deep_prompt_embeddings.shape[0]:
                deep_prompt_emb = self.prompt_dropout(
                    self.prompt_proj(
                        self.deep_prompt_embeddings[i - 1]).expand(
                            B, -1, -1)).permute(1, 0, 2)
                hidden_states = torch.cat(
                    (hidden_states[:1, :, :], 
                     deep_prompt_emb,
                     hidden_states[(1 + self.num_tokens):, :, :]),
                    dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)
            else:
                hidden_states = torch.cat(
                    (hidden_states[:1, :, :],
                     hidden_states[(1 + self.num_tokens):, :, :]),
                    dim=0)
                hidden_states = self.transformer.resblocks[i](hidden_states)

            if len(self.out_indices) > 1:
                if i in self.out_indices:
                    xp = hidden_states.permute(
                        1, 0, 2)[:,
                                 (1 + self.num_tokens):\
                                -sum(self.region_level_bridge_sizes), :].permute(
                                      0, 2, 1).reshape(B, -1, H, W)
                    features.append(xp.contiguous())

            if i == (self.num_layers - 2):
                before_last_feats = self.prompt_norm(hidden_states)

        encoded = torch.concat(
            (self.prompt_norm(hidden_states[:-sum(self.region_level_bridge_sizes), :, :]),
             hidden_states[-sum(self.region_level_bridge_sizes):, :, :]),
            dim=0)
        if out_last:
            return before_last_feats
        else:
            return encoded, features
