import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    def __init__(self, d_model, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, time, d_model)
        # 1. Pool over time: (batch, d_model, time) -> (batch, d_model, 1)
        y = self.avg_pool(x.transpose(-1, -2))
        # 2. Conv over channels: (batch, 1, d_model)
        y = self.conv(y.transpose(-1, -2)) # (batch, 1, d_model)
        y = self.sigmoid(y)
        # 3. Multiply: (batch, time, d_model) * (batch, 1, d_model)
        return x * y

class ALPE(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.mask_emb = nn.Embedding(2, d_model)

    def forward(self, x, mask):
        b, t, c = x.size()
        pos = self.pos_emb[:, :t, :]
        m_idx = mask.squeeze(-1).long()
        m_emb = self.mask_emb(m_idx)
        return pos + m_emb

class CNNSubmodule(nn.Module):
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU()
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)

class TransformerSubmodule(nn.Module):
    def __init__(self, d_model, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x+attn_out)

class CTFusion(nn.Module):
    def __init__(self, d_model, nhead, kernel_size):
        super().__init__()
        self.cnn = CNNSubmodule(d_model, kernel_size)
        self.transformer = TransformerSubmodule(d_model, nhead)
        self.eca = ECA(d_model)
        self.fusion = nn.Linear(d_model, d_model)

    def forward(self, x):
        c_out = self.cnn(x)
        t_out = self.transformer(x)
        out = c_out + t_out
        out = self.fusion(out)
        return self.eca(out)

class MCTNet(nn.Module):
    def __init__(self, input_dim=5, d_model=64, n_stages=3, nhead=4, kernel_size=3, num_classes=256):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.alpe = ALPE(d_model)
        self.stages = nn.ModuleList([CTFusion(d_model, nhead, kernel_size) for _ in range(n_stages)])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask):
        x = self.input_proj(x)
        pos = self.alpe(x, mask)
        x = x + pos
        for stage in self.stages:
            x = stage(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)