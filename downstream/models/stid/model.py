"""STID (Shao et al., CIKM 2022), adapted from https://github.com/zezhishao/STID."""

import torch
from torch import nn


class MLPBlock(nn.Module):
    """Residual block of two 1x1 convolutions with ReLU and dropout."""

    def __init__(self, input_dim, hidden_dim):
        """Create the two convolutions."""
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, x):
        """x + fc2(dropout(relu(fc1(x))))."""
        return self.fc2(self.drop(self.act(self.fc1(x)))) + x


class STID(nn.Module):
    """MLP encoder over series, node, time-of-day and day-of-week embeddings."""

    def __init__(
        self,
        num_nodes,
        input_len,
        output_len,
        steps_per_day,
        input_dim=3,
        embed_dim=32,
        node_dim=32,
        tid_dim=32,
        diw_dim=32,
        num_layer=3,
    ):
        """Create the embeddings, encoder and regression layer."""
        super().__init__()
        self.input_dim = input_dim
        self.steps_per_day = steps_per_day

        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(steps_per_day, tid_dim))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(7, diw_dim))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.series_emb = nn.Conv2d(input_dim * input_len, embed_dim, kernel_size=(1, 1), bias=True)
        hidden_dim = embed_dim + node_dim + tid_dim + diw_dim
        self.encoder = nn.Sequential(*[MLPBlock(hidden_dim, hidden_dim) for _ in range(num_layer)])
        self.regression = nn.Conv2d(hidden_dim, output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history):
        """Map history [B, H, V, 3] (value, time of day, day of week) to [B, P, V, 1]."""
        x = history[..., range(self.input_dim)]
        tid = history[:, -1, :, 1] * self.steps_per_day
        diw = history[:, -1, :, 2] * 7
        tid_emb = self.time_in_day_emb[tid.type(torch.LongTensor)]
        diw_emb = self.day_in_week_emb[diw.type(torch.LongTensor)]

        batch, _, nodes, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(batch, nodes, -1).transpose(1, 2).unsqueeze(-1)
        series = self.series_emb(x)
        node = self.node_emb.unsqueeze(0).expand(batch, -1, -1).transpose(1, 2).unsqueeze(-1)
        tid_emb = tid_emb.transpose(1, 2).unsqueeze(-1)
        diw_emb = diw_emb.transpose(1, 2).unsqueeze(-1)

        hidden = torch.cat([series, node, tid_emb, diw_emb], dim=1)
        return self.regression(self.encoder(hidden))
