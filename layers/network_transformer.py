import torch
from torch import nn

class NetworkTransformer(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, d_model=64, nhead=4, num_layers=2):
        super(NetworkTransformer, self).__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = d_model
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        # Patch Embedding
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Flatten Head
        self.flatten = nn.Flatten(start_dim=-2)
        self.fc_out = nn.Sequential(
            nn.Linear(self.patch_num * d_model, pred_len * 2),
            nn.GELU(),
            nn.Linear(pred_len * 2, pred_len)
        )


        # Linear Stream (trend)
        self.fc_trend1 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc_trend2 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        self.fc_trend3 = nn.Linear(pred_len // 2, pred_len)
        # Streams Concatination
        self.fc_concat = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s, t):
        # s: [Batch, Input, Channel] (seasonality)
        # t: [Batch, Input, Channel] (trend)
        s = s.permute(0,2,1) # [Batch, Channel, Input]
        t = t.permute(0,2,1)
        B, C, I = s.shape
        s = torch.reshape(s, (B*C, I)) # [Batch*Channel, Input]
        t = torch.reshape(t, (B*C, I))
        # Patching
        if self.padding_patch == 'end':
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [Batch*Channel, Patch_num, Patch_len]
        # Patch Embedding
        s = self.patch_embed(s) # [Batch*Channel, Patch_num, d_model]
        # Transformer Encoder
        s = self.transformer_encoder(s) # [Batch*Channel, Patch_num, d_model]
        # Flatten Head
        s = self.flatten(s) # [Batch*Channel, Patch_num*d_model]
        s = self.fc_out(s) # [Batch*Channel, pred_len]
        # Linear Stream (trend)
        t = self.fc_trend1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc_trend2(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc_trend3(t)
        # Streams Concatination
        x = torch.cat((s, t), dim=1) # [Batch*Channel, pred_len*2]
        x = self.fc_concat(x) # [Batch*Channel, pred_len]
        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # [Batch, Output, Channel]
        return x
