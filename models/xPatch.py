import torch
import torch.nn as nn
import math

from layers.decomp import DECOMP
from layers.network import Network
# from layers.net_linear import Network
# from layers.network_mlp import NetworkMLP # For ablation study with MLP-only stream
# from layers.network_cnn import NetworkCNN # For ablation study with CNN-only stream
from layers.revin import RevIN
import matplotlib.pyplot as plt

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # Parameters
        seq_len = configs.seq_len   # lookback window L
        pred_len = configs.pred_len # prediction length (96, 192, 336, 720)
        c_in = configs.enc_in       # input channels
        d_model = configs.d_model    # dimension of model
        period_len = configs.period_len  # period length
        nhead = configs.n_head      # number of attention heads

        # Patching
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch

        # Normalization
        self.revin = configs.revin
        self.revin_layer = RevIN(c_in,affine=True,subtract_last=False)

        # Moving Average
        self.ma_type = configs.ma_type
        alpha = configs.alpha       # smoothing factor for EMA (Exponential Moving Average)
        beta = configs.beta         # smoothing factor for DEMA (Double Exponential Moving Average)

        dropout = configs.dropout
        num_layers = configs.num_layers

        self.decomp = DECOMP(self.ma_type, alpha, beta, period_len)
        # self.net = Network(seq_len, pred_len, c_in, period_len, d_model, dropout)
        # self.net = Network(seq_len, pred_len, c_in, period_len, d_model)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch, dropout, d_model, nhead, num_layers)
        # self.net_mlp = NetworkMLP(seq_len, pred_len) # For ablation study with MLP-only stream
        # self.net_cnn = NetworkCNN(seq_len, pred_len, patch_len, stride, padding_patch) # For ablation study with CNN-only stream

    def forward(self, x):
        # x: [Batch, Input, Channel]


        # x_120 = x[1, :120, 6]  # Lấy batch đầu tiên, 120 bước, channel đầu tiên

        # param_list = [
        #     (0.2, 0.2),
        #     (0.4, 0.4),
        #     (0.6, 0.6),
        #     (0.7, 0.7),
        #     (0.8, 0.8),
        #     (0.9, 0.9)
        # ]

        # # Sau đó truyền x_120 vào các hàm decomp (cần đảm bảo shape phù hợp, thường là [1, 120, 1])
        # x_120 = x_120.unsqueeze(0).unsqueeze(-1)  # shape: [1, 120, 1]

        # decomp_sma = DECOMP(ma_type='sma', alpha=0.5, beta=0.3, window_size=25)
        # decomp_dema = DECOMP(ma_type='dema', alpha=0.3, beta=0.3, window_size=25)

        # sma_resid, sma_trend = decomp_sma(x_120)
        # dema_seasonal, dema_trend = decomp_dema(x_120)

        # # Chuyển sang numpy để vẽ
        # data_np = x_120.squeeze().detach().cpu().numpy()
        # sma_trend = sma_trend.squeeze().detach().cpu().numpy()
        # sma_resid = sma_resid.squeeze().detach().cpu().numpy()
        # dema_trend = dema_trend.squeeze().detach().cpu().numpy()
        # dema_seasonal = dema_seasonal.squeeze().detach().cpu().numpy()

        # # fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
        # # axes = axes.flatten()
        # # plt.plot(range(120), data_np, color='#3abeb6', label='RAW')
        # # for i, (alpha, beta) in enumerate(param_list):
        # #     decomp_dema = DECOMP(ma_type='dema', alpha=alpha, beta=beta, window_size=25)
        # #     dema_seasonal, dema_trend = decomp_dema(x_120)
        # #     dema_trend = dema_trend.squeeze().detach().cpu().numpy()
        # #     axes[i].plot(range(120), data_np, color='#3abeb6', label='RAW')
        # #     axes[i].plot(range(120), dema_trend, color='#d70527', label=f'DEMA a={alpha}, b={beta}')
        # #     axes[i].set_title(f'DEMA a={alpha}, b={beta}')
        # #     axes[i].set_ylabel('120')
        # #     axes[i].legend()
        # # plt.tight_layout()
        # # plt.savefig('dema_trend_grid.png', bbox_inches='tight', dpi=300)
        # # plt.close()


        # # fig, axes = plt.subplots(2, 3, figsize=(18, 8), sharex=True, sharey=True)
        # # axes = axes.flatten()
        # # plt.plot(range(120), data_np, color='#3abeb6', label='RAW')
        # # for i, (alpha, beta) in enumerate(param_list):
        # #     decomp_dema = DECOMP(ma_type='dema', alpha=alpha, beta=beta, window_size=25)
        # #     dema_seasonal, dema_trend = decomp_dema(x_120)
        # #     dema_trend = dema_trend.squeeze().detach().cpu().numpy()
        # #     dema_seasonal = dema_seasonal.squeeze().detach().cpu().numpy()
        # #     axes[i].plot(range(120), data_np, color='#3abeb6', label='RAW')
        # #     axes[i].plot(range(120), dema_seasonal, color='#d70527', label=f'Seasonal a={alpha}, b={beta}')
        # #     axes[i].set_title(f'Seasonal a={alpha}, b={beta}')
        # #     axes[i].set_ylabel('120')
        # #     axes[i].legend()

        # # plt.tight_layout()
        # # plt.savefig('dema_seasonal_grid.png', bbox_inches='tight', dpi=300)
        # # plt.close()


        # # Vẽ hình
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 6))
        # plt.plot(data_np, color='#3abeb6', label='Data')
        # plt.plot(sma_trend, color='#d70527', label='Trend')
        # plt.plot(sma_resid, color='#3334c9', label='Seasonality')
        # plt.legend()
        # plt.ylabel('120', fontsize=16)
        # # plt.title('So sánh Trend của SMA, DEMA với chuỗi gốc')
        # # plt.show()
        # plt.savefig('ETTh1_sma.png', bbox_inches='tight', dpi=300)


        # plt.figure(figsize=(12, 6))
        # plt.plot(data_np, color='#3abeb6', label='Data')
        # plt.plot(dema_trend, color='#d70527', label='Trend')
        # plt.plot(dema_seasonal, color='#3334c9', label='Seasonality')
        # plt.legend()
        # plt.ylabel('120', fontsize=16)
        # # plt.title('So sánh Trend của SMA, DEMA với chuỗi gốc')
        # # plt.show()
        # plt.savefig('ETTh1_dema.png', bbox_inches='tight', dpi=300)
        # print("1")
        # return

        # Normalization
        if self.revin:
            x = self.revin_layer(x, 'norm')


        if self.ma_type == 'reg':   # If no decomposition, directly pass the input to the network
            x = self.net(x, x)
            # x = self.net_mlp(x) # For ablation study with MLP-only stream
            # x = self.net_cnn(x) # For ablation study with CNN-only stream
        if self.ma_type == 'stl':
            seasonal_init, trend_init, resid_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init, resid_init)
        if self.ma_type == 'sma':  
            resid_init, trend_init = self.decomp(x)
            x = self.net(resid_init, trend_init)
        if self.ma_type == 'dema':
            seasonal_init, trend_init = self.decomp(x)
            x = self.net(seasonal_init, trend_init)

        # Denormalization
        if self.revin:
            x = self.revin_layer(x, 'denorm')

        return x