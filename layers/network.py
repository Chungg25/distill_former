import torch
from torch import nn
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from fastdtw import fastdtw

class Network(nn.Module):
    def __init__(self, seq_len, pred_len, patch_len, stride, padding_patch, d_model=64, nhead=4, num_layers=2, n_clusters=3):
        super(Network, self).__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = d_model
        self.patch_num = (seq_len - patch_len)//stride + 1
        if padding_patch == 'end':
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1
        self.n_clusters = n_clusters
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
            nn.Linear(self.n_clusters * d_model, pred_len * 2),
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

    def cluster_patches(self, patches, dtw_threshold=5.0):
        # patches: [N_patch, patch_len] (numpy)
        N = patches.shape[0]
        groups = []
        current_group = [0]
        for i in range(1, N):
            dist, _ = fastdtw(patches[i], patches[i-1])
            if dist <= dtw_threshold:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]
        if current_group:
            groups.append(current_group)
        # Chọn patch đại diện cho mỗi nhóm (gộp)
        representatives = []
        for group in groups:
            group_patches = patches[group]
            centroid = np.mean(group_patches, axis=0)
            min_dist = float('inf')
            rep_idx = group[0]
            for idx in group:
                d, _ = fastdtw(patches[idx], centroid)
                if d < min_dist:
                    min_dist = d
                    rep_idx = idx
            representatives.append(rep_idx)
        return representatives

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
        s_patches = s.unfold(dimension=-1, size=self.patch_len, step=self.stride) # [Batch*Channel, Patch_num, Patch_len]
        # Patch Embedding
        s_emb = self.patch_embed(s_patches) # [Batch*Channel, Patch_num, d_model]
        # DTW clustering for each sample in batch*channel
        reps = []
        for i in range(s_patches.shape[0]):
            patches_np = s_patches[i].detach().cpu().numpy()
            rep_idx = self.cluster_patches(patches_np)
            reps.append(s_emb[i, rep_idx]) # [n_clusters, d_model]
        s_reps = torch.stack(reps, dim=0) # [Batch*Channel, n_clusters, d_model]
        # Transformer Encoder
        s_out = self.transformer_encoder(s_reps) # [Batch*Channel, n_clusters, d_model]
        # Flatten Head
        s_out = self.flatten(s_out) # [Batch*Channel, n_clusters*d_model]
        s_out = self.fc_out(s_out) # [Batch*Channel, pred_len]
        # Linear Stream (trend)
        t = self.fc_trend1(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc_trend2(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc_trend3(t)
        # Streams Concatination
        x = torch.cat((s_out, t), dim=1) # [Batch*Channel, pred_len*2]
        x = self.fc_concat(x) # [Batch*Channel, pred_len]
        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len)) # [Batch, Channel, Output]
        x = x.permute(0,2,1) # [Batch, Output, Channel]
        return x
