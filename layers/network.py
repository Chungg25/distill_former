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
            groups=in_channels,
            # padding=0,
            # dilation=dilation
        )

    def forward(self, x):
        # x: [B, C, T]
        pad = (self.kernel_size - 1) * self.dilation
        x = F.pad(x, (pad, 0))   # pad LEFT only (causal)
        return self.conv(x)


class AdaptiveFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.fc = nn.Linear(pred_len, pred_len)

    def forward(self, s, t):
        alpha = torch.sigmoid(self.fc(s + t))
        return alpha * s + (1 - alpha) * t


class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch,
                 dropout=0.1, d_model=64, nhead=4, num_layers=2):
        super().__init__()

        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch

        patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        self.patch_num = patch_num

        self.patch_glu = PatchChannelGLU(patch_len, d_model)


        self.patch_embed = nn.Linear(d_model, d_model)

        self.patch_conv = CausalConv1d(d_model, d_model, kernel_size=3, dilation=1)
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
            num_layers=num_layers
        )


        self.flatten = nn.Flatten(start_dim=-2)
        # self.patch_fc = nn.Sequential(
        #     nn.Linear(self.patch_num * d_model, pred_len * 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(pred_len * 2, pred_len)
        # )

        self.fl_linear = nn.Linear(self.patch_num * d_model, pred_len * 2)
        self.fl_gelu = nn.GELU()
        self.fl_dropout = nn.Dropout(dropout)
        self.fl_linear2 = nn.Linear(pred_len * 2, pred_len)

        # self.fc_trend = nn.Sequential(
        #     nn.Linear(seq_len, pred_len * 2),
        #     nn.AvgPool1d(kernel_size=2),
        #     nn.LayerNorm(pred_len),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(pred_len, pred_len)
        # )

        self.fc_trend2 = nn.Linear(seq_len, pred_len * 2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len)
        self.gelu2 = nn.GELU()
        self.fc_dropout2 = nn.Dropout(dropout)
        self.fc_trend3 = nn.Linear(pred_len, pred_len)

        # self.fc_trend = nn.Sequential(
        #     nn.Linear(seq_len, pred_len * 2),
        #     nn.GELU(),
        #     nn.Linear(pred_len * 2, pred_len)
        # )

        
        self.adaptive_fusion = AdaptiveFusion(pred_len)
        # self.fc = nn.Linear(pred_len*2, pred_len)

    def forward(self, s, t):
        # s, t: [B, seq_len, C]
        s = s.permute(0, 2, 1)  # [B, C, seq_len]
        t = t.permute(0, 2, 1)
        B, C, I = s.shape

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

        s_patch = s_patch.permute(0, 2, 1)    # [B*C, d_model, patch_num]
        s_patch = self.patch_conv(s_patch)
        s_patch = s_patch.permute(0, 2, 1)    # [B*C, new_patch_num, d_model]


        s_patch_residual = s_patch
        # mask = torch.triu(torch.ones(s_patch.shape[1], s_patch.shape[1], device=s.device), diagonal=1).bool()
        # s_patch = self.transformer_encoder(s_patch, mask=mask)
        s_patch = self.transformer_encoder(s_patch)
        s_patch = s_patch + s_patch_residual

        s_patch = self.flatten(s_patch)
        # s_out = self.patch_fc(s_patch)        # [B*C, pred_len]
        s = self.fl_linear(s_patch)
        s = self.fl_gelu(s)
        s = self.fl_dropout(s)
        s = self.fl_linear2(s)
        # s = self.fl_dropout(s)

        t = t.reshape(B * C, I)
        # t_out = self.fc_trend(t_flat)

        t = self.fc_trend2(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.gelu2(t)
        t = self.fc_dropout2(t)
        t = self.fc_trend3(t)

        # x = torch.cat((s_out, t_out), dim=1)
        # x = self.fc(x)
        # x = torch.reshape(x, (B, C, self.pred_len))
        # x = x.permute(0, 2, 1)  


        x = self.adaptive_fusion(s, t)
        x = x.view(B, C, self.pred_len)
        x = x.permute(0, 2, 1)                # [B, pred_len, C]

        # x = s_out + t_out
        # x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        # x = x.permute(0,2,1)

        return x
