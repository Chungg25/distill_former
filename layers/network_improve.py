import torch
from torch import nn

class PatchChannelGLU(nn.Module):
    def __init__(self, patch_len, d_model):
        super().__init__()
        self.linear_a = nn.Linear(patch_len, d_model)
        self.linear_b = nn.Linear(patch_len, d_model)

    def forward(self, x):
        a = self.linear_a(x)
        b = torch.sigmoid(self.linear_b(x))
        return a * b


class WeightedPatchWiseHead(nn.Module):
    def __init__(self, d_model, pred_len):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, pred_len)
        )
        self.weight = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, 1)
        )

        # self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)

        self.res_proj = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # attn_out, _ = self.attn(x, x, x)   # self-attention giữa các patch
        y = self.proj(x)            # dự đoán patch
        w = self.weight(x).softmax(1)  # trọng số patch
        out = (y * w).sum(dim=1) + self.res_proj(x.mean(dim=1))  # weighted sum + residual
        return out

class AdaptiveFusion(nn.Module):
    def __init__(self, pred_len):
        super().__init__()
        self.fc = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # s, t: [B*C, pred_len]
        alpha = torch.sigmoid(self.fc(torch.cat([s, t], dim=-1)))
        return alpha * s + (1 - alpha) * t

class channel_transformer_block(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # seq_len = x.shape[1]  # channels dimension
        # mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer(x)
        return self.norm(out + x)

class auto_transformer_block(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,  # [B, T, D]
            dim_feedforward=d_model*2,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, C, d_model] -> treat C as sequence length
        seq_len = x.shape[1]
        # Causal mask: prevent attention to future channels
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device) * float('-inf'), diagonal=1)
        out = self.transformer(x, mask=mask)
        return self.norm(out + x)
class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, droupout = 0, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.d_model = d_model


        # Patch number
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # --- Patch-level ---
        self.patch_channel_glu = PatchChannelGLU(patch_len, d_model)
        self.patch_embed = nn.Linear(d_model, d_model)
        self.auto_attn_block = auto_transformer_block(d_model=d_model, dropout=droupout)
        self.patch_repr = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        # self.flatten = nn.Flatten(start_dim=-2)
        # self.fc_patch = nn.Sequential(
        #     nn.Linear(self.patch_num*d_model, pred_len),
        #     nn.GELU(),
        #     nn.Dropout(0.1)
        # )

        self.patch_head = WeightedPatchWiseHead(d_model, pred_len)

        self.channel_proj = nn.Linear(pred_len, d_model)
        self.channel_repr = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        # --- Channel-level (on seasonal) ---
        self.channel_attn_block = channel_transformer_block(d_model=d_model, dropout=droupout)
        self.channel_fc = nn.Linear(d_model, pred_len)

        # --- Trend ---
        self.fc_trend = nn.Linear(seq_len, pred_len*2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len)
        self.fc_trend3 = nn.Linear(pred_len, pred_len)

        # self.adaptive_fusion = AdaptiveFusion(pred_len)

    def forward(self, s, t):
        B, I, C = s.shape[0], s.shape[1], s.shape[2]
        s = s.permute(0,2,1).reshape(B*C, I)
        t = t.permute(0,2,1).reshape(B*C, I)

        # --- Patch-level on seasonal ---
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(-1, self.patch_len, self.stride)
        s = self.patch_channel_glu(s)
        s = self.patch_embed(s)
        s = self.auto_attn_block(s)
        s = self.patch_repr(s)
        # s = self.flatten(s)
        # s = self.fc_patch(s)
        s = self.patch_head(s)
        s = s.reshape(B, C, self.pred_len)

        # --- Channel-level attention on seasonal ---
        s = self.channel_proj(s)
        # print(s.shape)   
        s = self.channel_attn_block(s)
        s = self.channel_repr(s)
        s = self.channel_fc(s)

        # --- Trend (simple linear) ---
        t = self.fc_trend(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc_trend3(t)
        t = t.reshape(B, C, self.pred_len)
        
        # print(s.shape, t.shape)
        # x = self.adaptive_fusion(s, t)
        # x = x.view(B, C, self.pred_len)

        out = s + t
        return out.permute(0,2,1)

        # --- Combine ---
        # x = self.adaptive_fusion(s, t)
        # x = x.view(B, C, self.pred_len)
        x = x.permute(0, 2, 1) # [Batch, Output, Channel]
        return x