import torch
from torch import nn
import math


class ChannelTimeAttention(nn.Module):
    """
    Attention across variables (channels) and across time steps, merged.
    Input:  y in shape [B, C, L]
    Output: same shape [B, C, L]
    """

    def __init__(self, dropout: float = 0.0, residual: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # weight for channel attention view
        self.beta = nn.Parameter(torch.tensor(0.5))   # weight for temporal attention view
        self.gamma = nn.Parameter(torch.tensor(0.0))  # residual mixing
        self.residual = residual
        self.drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B, C, L]
        B, C, L = y.shape

        # Channel-wise attention (tokens=C, features=L)
        # scores_c: [B, C, C]
        scale_c = 1.0 / math.sqrt(max(L, 1))
        scores_c = torch.matmul(y, y.transpose(-1, -2)) * scale_c
        attn_c = torch.softmax(scores_c, dim=-1)
        y_c = torch.matmul(attn_c, y)  # [B, C, L]

        # Temporal attention (tokens=L, features=C)
        yt = y.transpose(1, 2)  # [B, L, C]
        scale_t = 1.0 / math.sqrt(max(C, 1))
        scores_t = torch.matmul(yt, yt.transpose(-1, -2)) * scale_t  # [B, L, L]
        attn_t = torch.softmax(scores_t, dim=-1)
        y_t = torch.matmul(attn_t, yt).transpose(1, 2)  # [B, C, L]

        out = self.alpha * y_c + self.beta * y_t
        if self.residual:
            out = out + self.gamma * y
        return self.drop(out)


class Network(nn.Module):
    """
    Dual-attention forecaster:
    - Keep seasonal and trend streams as inputs.
    - Apply attention across variables and across time for each stream.
    - Project time dimension from seq_len -> pred_len.
    - Fuse seasonal and trend for final prediction.

    Expected input: s, t with shape [B, seq_len, C]
    Output: [B, pred_len, C]
    """

    def __init__(self, seq_len, pred_len, c_in,
                 period_len=None, d_model=None, dropout: float = 0.0,
                 patch_len: int = None, stride: int = None, padding_patch: str = None):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.c_in = c_in

        # Patching settings
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.use_patching = (patch_len is not None) and (stride is not None)

        if self.use_patching:
            # number of patches (same formula as original implementation)
            self.patch_num = (seq_len - patch_len) // stride + 1
            if padding_patch == 'end':
                # replicate pad with stride length (simple variant)
                self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
                self.patch_num += 1
            s_proj_in = self.patch_num
        else:
            self.patch_num = None
            s_proj_in = seq_len

        # Attention block for seasonal; trend will use linear only as requested
        self.s_attn = ChannelTimeAttention(dropout=dropout, residual=True)
        # self.t_attn is intentionally omitted (trend uses linear only)

        # Project along time dimension: tokens -> pred_len, per channel
        # Seasonal may be patched, trend stays on original sequence length
        self.s_time_proj = nn.Linear(s_proj_in, pred_len)
        self.t_time_proj = nn.Linear(seq_len, pred_len)

        # Optional normalization
        self.s_norm = nn.LayerNorm([c_in, pred_len])
        self.t_norm = nn.LayerNorm([c_in, pred_len])

        # Fuse seasonal and trend (concat channels then linear)
        self.fuse = nn.Linear(2 * c_in, c_in)

    def forward(self, s: torch.Tensor, t: torch.Tensor, resid: torch.Tensor = None) -> torch.Tensor:
        # Inputs: [B, L, C]
        B, L, C = s.shape

        # To [B, C, L] for attention / patching
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        if self.use_patching:
            # Optional padding at end to allow an extra patch (seasonal only)
            if self.padding_patch == 'end':
                s = self.padding_patch_layer(s)
            # Unfold seasonal into patches: [B, C, patch_num, patch_len]
            s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # Reduce patch dimension (mean) -> [B, C, patch_num]
            s = s.mean(dim=-1)

        # Attention over seasonal (patched or not) temporal tokens
        s = self.s_attn(s)  # [B, C, (L or patch_num)]
        # Trend path: linear only (no attention, no patching)

        # Time projection (tokens -> pred_len)
        s = self.s_time_proj(s)
        t = self.t_time_proj(t)

        # Optional norm (over last two dims per batch)
        s = self.s_norm(s)
        t = self.t_norm(t)

        # Back to [B, pred_len, C]
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        # Fuse seasonal + trend
        x = torch.cat([s, t], dim=-1)  # [B, pred_len, 2C]
        x = self.fuse(x)               # [B, pred_len, C]

        return x