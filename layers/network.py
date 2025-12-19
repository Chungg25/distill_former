import torch
from torch import nn
import torch.nn.functional as F

# ===== Patch-level GLU =====
class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):  # x: [B*C, patch_num, patch_len]
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            dilation=dilation
        )

    def forward(self, x):
        # x: [B, C, T]
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))   # pad LEFT only (causal)
        return self.conv(x)


class PatchTimeEmbedding(nn.Module):
    def __init__(self, max_patch_num, d_model):
        super().__init__()
        self.embedding = nn.Embedding(max_patch_num, d_model)

    def forward(self, x):
        # x: [B*C, P, D]
        P = x.size(1)
        pos = torch.arange(P, device=x.device).unsqueeze(0)
        return x + self.embedding(pos)



class AdaptiveFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.fc = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        alpha = torch.sigmoid(self.fc(torch.cat([s, t], dim=-1)))
        return alpha * s + (1 - alpha) * t

class GatedPatchFC(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc_a = nn.Linear(in_dim, out_dim)
        self.fc_b = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        a = self.fc_a(x)
        b = torch.sigmoid(self.fc_b(x))
        return self.dropout(a * b)



# ===== Full Network =====
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, d_model=64, nhead=4):
        super().__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        # ---- Patch number ----
        patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Sau pooling stride=2
        self.patch_num = patch_num // 2

        # ---- Patch-level ----
        self.patch_glu = PatchChannelGLU(patch_len, d_model)
        self.patch_embed = nn.Linear(d_model, d_model)

        # 🔥 Conv1d + Pooling (NEW)
        # self.patch_conv = nn.Sequential(
        #     CausalConv1d(d_model, d_model, kernel_size=3, dilation=1),
        #     nn.GELU(),
        #     CausalConv1d(d_model, d_model, kernel_size=3, dilation=2),
        #     nn.GELU(),
        # )
        # self.patch_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # self.patch_conv = nn.Conv1d(
        #     in_channels=d_model,
        #     out_channels=d_model,
        #     kernel_size=3,
        #     padding=1
        # )
        self.patch_conv = CausalConv1d(d_model, d_model, kernel_size=3, dilation=1)
        self.patch_pool = nn.AvgPool1d(kernel_size=2, stride=2)

        # self.patch_time_emb = PatchTimeEmbedding(
        #     max_patch_num=self.patch_num,
        #     d_model=d_model
        # )


        # ---- Transformer ----
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 2,
                # dim_feedforward = 1024,
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ),
            num_layers=2
        )

        self.flatten = nn.Flatten(start_dim=-2)
        self.patch_fc = nn.Sequential(
            # GatedPatchFC(self.patch_num * d_model, pred_len * 2, dropout),
            nn.Linear(self.patch_num * d_model, pred_len * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_len * 2, pred_len)
        )

        # self.patch_fc = GatedPatchHead(self.patch_num, d_model, pred_len, dropout)

        # ---- Trend stream ----
        self.fc_trend = nn.Sequential(
            nn.Linear(seq_len, pred_len * 4),
            nn.AvgPool1d(kernel_size=2),
            nn.LayerNorm(pred_len*2),
            nn.Dropout(dropout),
            nn.Linear(pred_len*2, pred_len)
        )

        # self.trend_encoder = MultiScaleTrend(
        #     seq_len=seq_len,
        #     pred_len=pred_len,
        #     hidden=pred_len * 4,
        #     dropout=dropout
        # )


        # self.adaptive_fusion = AdaptiveFusion(pred_len)

    def forward(self, s, t):
        # s, t: [B, seq_len, C]
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        t = t.permute(0, 2, 1)
        B, C, I = s.shape

        # ---- Patch-level ----
        s_flat = s.reshape(B * C, I)
        if self.padding_patch == 'end':
            s_flat = self.padding_patch_layer(s_flat)

        s_patch = s_flat.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.stride
        )  # [B*C, patch_num, patch_len]

        s_patch = self.patch_glu(s_patch)     # [B*C, patch_num, d_model]
        s_patch = self.patch_embed(s_patch)

        # 🔥 Conv1d + Pooling
        s_patch = s_patch.permute(0, 2, 1)    # [B*C, d_model, patch_num]
        s_patch = self.patch_conv(s_patch)
        s_patch = self.patch_pool(s_patch)
        s_patch = s_patch.permute(0, 2, 1)    # [B*C, new_patch_num, d_model]

        # ---- Transformer ----
        # s_patch = self.patch_time_emb(s_patch)
        s_patch = self.transformer_encoder(s_patch)

        s_patch = self.flatten(s_patch)
        s_out = self.patch_fc(s_patch)        # [B*C, pred_len]

        # ---- Trend stream ----
        t_flat = t.reshape(B * C, I)
        t_out = self.fc_trend(t_flat)
        # t_out = self.trend_encoder(t_flat)

        # ---- Fusion ----
        # x = self.adaptive_fusion(s_out, t_out)
        # x = x.view(B, C, self.pred_len)
        # x = x.permute(0, 2, 1)                # [B, pred_len, C]

        x = s_out + t_out
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        x = x.permute(0,2,1)

        return x
