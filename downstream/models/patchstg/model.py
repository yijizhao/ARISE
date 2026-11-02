"""PatchSTG (Fang et al., KDD 2025), adapted from https://github.com/LMissher/PatchSTG."""

import torch
from timm.models.vision_transformer import Attention, Mlp
from torch import nn


class WindowAttBlock(nn.Module):
    """Attention within spatial patches (depth) and across patches (breadth)."""

    def __init__(self, hidden_size, num_heads, num, size, mlp_ratio=4.0):
        """Create the two attention and MLP sub-blocks."""
        super().__init__()
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size
        self.nnorm1 = nn.LayerNorm(hidden_size)
        self.nattn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1
        )
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden, act_layer=nn.GELU, drop=0.1
        )
        self.snorm1 = nn.LayerNorm(hidden_size)
        self.sattn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1
        )
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden, act_layer=nn.GELU, drop=0.1
        )

    def forward(self, x):
        """Apply depth attention, then breadth attention, on [B, T, nodes, D]."""
        B, T, _, D = x.shape
        P, N = self.num, self.size
        x = x.reshape(B, T, P, N, D)
        qkv = self.snorm1(x.reshape(B * T * P, N, D))
        x = x + self.sattn(qkv).reshape(B, T, P, N, D)
        x = x + self.smlp(self.snorm2(x))
        qkv = self.nnorm1(x.transpose(2, 3).reshape(B * T * N, P, D))
        x = x + self.nattn(qkv).reshape(B, T, N, P, D).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))
        return x.reshape(B, T, -1, D)


class PatchSTG(nn.Module):
    """Spatial-patch transformer over a KD-tree ordering of the nodes."""

    def __init__(
        self,
        tem_patchsize,
        tem_patchnum,
        node_num,
        spa_patchsize,
        spa_patchnum,
        tod,
        dow,
        layers,
        factors,
        input_dims,
        node_dims,
        tod_dims,
        dow_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
    ):
        """Create the input embeddings, attention blocks and output layer."""
        super().__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow
        dims = input_dims + tod_dims + dow_dims + node_dims

        self.input_st_fc = nn.Conv2d(
            3, input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True
        )
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.spa_encoder = nn.ModuleList(
            [
                WindowAttBlock(
                    dims, 1, spa_patchnum // factors, spa_patchsize * factors, mlp_ratio=1
                )
                for _ in range(layers)
            ]
        )
        self.regression_conv = nn.Conv2d(
            tem_patchnum * dims, tem_patchsize * tem_patchnum, kernel_size=(1, 1), bias=True
        )

    def forward(self, x, te):
        """Map x [B, H, V, 1] and integer calendar te [B, H, V, 2] to [B, P, V, 1]."""
        x = self.embedding(x, te)
        x = x[:, :, self.reo_all_idx, :]
        for block in self.spa_encoder:
            x = block(x)
        original = torch.zeros(x.shape[0], x.shape[1], self.node_num, x.shape[-1]).to(x.device)
        original[:, :, self.ori_parts_idx, :] = x[:, :, self.reo_parts_idx, :]
        original = original.transpose(2, 3).reshape(original.shape[0], -1, original.shape[-2], 1)
        return self.regression_conv(original)

    def embedding(self, x, te):
        """Concatenate the patch embedding with time-of-day, day-of-week and node embeddings."""
        b = x.shape[0]
        x1 = torch.cat([x, te[..., 0:1] / self.tod, te[..., 1:2] / self.dow], -1).float()
        data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        t = data.shape[1]
        tid = te[:, -t:, :, 0]
        data = torch.cat([data, self.time_in_day_emb[tid.type(torch.LongTensor)]], -1)
        diw = te[:, -t:, :, 1]
        data = torch.cat([data, self.day_in_week_emb[diw.type(torch.LongTensor)]], -1)
        node = self.node_emb.unsqueeze(0).unsqueeze(1).expand(b, t, -1, -1)
        return torch.cat([data, node], -1)
