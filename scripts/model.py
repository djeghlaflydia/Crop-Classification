"""
model.py - MCTNet: CNN-Transformer hybrid for crop classification
Based on "A lightweight CNN-Transformer network for pixel-based crop mapping"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA(nn.Module):
    """Efficient Channel Attention module"""
    def __init__(self, d_model, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, time, d_model)
        y = self.avg_pool(x.transpose(-1, -2))
        y = self.conv(y.transpose(-1, -2))
        y = self.sigmoid(y)
        return x * y

class ALPE(nn.Module):
    """Adaptive Learned Positional Encoding with mask"""
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
    """CNN submodule for local feature extraction"""
    def __init__(self, d_model, kernel_size=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)

class TransformerSubmodule(nn.Module):
    """Transformer submodule for global dependencies"""
    def __init__(self, d_model, nhead=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + self.dropout(attn_out))

class CTFusion(nn.Module):
    """CNN-Transformer Fusion module"""
    def __init__(self, d_model, nhead, kernel_size, dropout=0.1):
        super().__init__()
        self.cnn = CNNSubmodule(d_model, kernel_size)
        self.transformer = TransformerSubmodule(d_model, nhead, dropout)
        self.eca = ECA(d_model)
        self.fusion = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        c_out = self.cnn(x)
        t_out = self.transformer(x)
        out = c_out + t_out
        out = self.fusion(out)
        out = self.dropout(out)
        out = self.norm(out + x)
        return self.eca(out)

class MCTNet(nn.Module):
    """Multi-stage CNN-Transformer Network for crop classification"""
    def __init__(self, input_dim=7, d_model=64, n_stages=3, nhead=4, 
                 kernel_size=3, num_classes=256, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        self.alpe = ALPE(d_model)
        
        self.stages = nn.ModuleList([
            CTFusion(d_model, nhead, kernel_size, dropout) 
            for _ in range(n_stages)
        ])
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, x, mask):
        """
        Args:
            x: (batch, time, features) - input time series
            mask: (batch, time, 1) - validity mask (1=valid, 0=missing)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.input_proj(x)
        pos = self.alpe(x, mask)
        x = x + pos
        
        for stage in self.stages:
            x = stage(x)
        
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)