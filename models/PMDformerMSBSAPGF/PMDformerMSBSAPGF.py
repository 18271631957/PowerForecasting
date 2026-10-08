import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Transformer_Enc import Encoder_TRA, EncoderLayer_TRA, Encoder_PVA, EncoderLayer_PVA
from .layers.SelfAttention_Family import TrendRestorationAttention, SelfAttention

from einops import rearrange


class BPA_Fusion(nn.Module):
    def __init__(self, valid_nwp_indices, d_mark, d_model, n_heads=4, dropout=0.1):
        super(BPA_Fusion, self).__init__()
        self.register_buffer('valid_idx', torch.tensor(valid_nwp_indices, dtype=torch.long))

        self.nwp_dim = len(valid_nwp_indices) + d_mark

        self.hist_proj = nn.Linear(2, d_model)
        self.nwp_proj = nn.Linear(self.nwp_dim, d_model)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)

        self.gate_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        self.fusion_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.zenith_envelope = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        self.zenith_bias = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )

        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, hist_pred, seq_future, seq_y_mark):
        zenith_future = seq_future[:, :, -1:]
        nwp_future = torch.index_select(seq_future, dim=-1, index=self.valid_idx)

        exo_input = torch.cat([nwp_future, seq_y_mark], dim=-1)

        h_emb = self.hist_proj(hist_pred)
        q_nwp = self.nwp_proj(exo_input)

        attn_out, _ = self.cross_attn(query=q_nwp, key=h_emb, value=h_emb)
        calibrated_nwp = self.norm1(q_nwp + attn_out)

        gate = self.gate_layer(torch.cat([calibrated_nwp, h_emb], dim=-1))
        fused_state = gate * calibrated_nwp + (1 - gate) * h_emb
        fused_state = self.norm2(fused_state + self.fusion_ffn(fused_state))

        solar_mask = self.zenith_envelope(zenith_future)
        solar_shift = self.zenith_bias(zenith_future)
        phys_modulated = fused_state * solar_mask + solar_shift

        out = self.out_proj(phys_modulated) + hist_pred[:, :, 0:1]
        return out


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.e_layers = configs.e_layers
        self.RevIn = configs.RevIn
        self.patch_size = configs.patch_size
        self.stride = configs.patch_size

        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.Embedding_layer = nn.Linear(self.patch_size, configs.d_model)
        self.Predicting_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)

        self.Encoder_TRA = Encoder_TRA(
            [
                EncoderLayer_TRA(
                    TrendRestorationAttention(d_model=configs.d_model, n_heads=configs.n_heads,
                                              dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.Encoder_PVA = Encoder_PVA(
            [
                EncoderLayer_PVA(
                    SelfAttention(d_model=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout
                ) for l in range(configs.v_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.patch_pos = nn.Parameter(torch.zeros(self.patch_num, configs.d_model))

        self.bpa_fusion = BPA_Fusion(
            valid_nwp_indices=configs.valid_nwp_indices,
            d_mark=configs.d_mark,
            d_model=configs.d_model,
            n_heads=configs.n_heads,
            dropout=configs.dropout
        )

    def forward(self, seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future, mask=None):
        # print(seq_future[:, :, -2:-1])
        x = seq_x
        B, L, N = x.shape

        x = rearrange(x, 'b l n -> b n l')
        if self.RevIn:
            means = x.mean(-1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b n p s -> (b n) p s')

        patch_means = x.mean(-1, keepdim=True)
        x = x - patch_means

        x = self.Embedding_layer(x)
        x = x + self.patch_pos.unsqueeze(0)

        x_main = x[:, :-1, :]
        x_last = x[:, -1:, :]
        x_last = rearrange(x_last, '(b n) p d -> (b p) n d', b=B, n=N)
        x_last, _ = self.Encoder_PVA(x_last)
        x_last = rearrange(x_last, '(b p) n d -> (b n) p d', b=B, n=N)
        x = torch.cat([x_main, x_last], dim=1)

        x, _ = self.Encoder_TRA(x, patch_means)
        x = x + patch_means
        x = rearrange(x, '(b n) p d -> b n (p d)', b=B, n=N)
        x = self.Predicting_layer(x)

        if self.RevIn:
            x = x * stdev + means

        x = rearrange(x, 'b n t -> b t n')

        # 同时提取功率与天顶角的内生预测
        hist_pred = x[:, :, -2:]

        out = self.bpa_fusion(
            hist_pred=hist_pred,
            seq_future=seq_future[:, -self.pred_len:, :],
            seq_y_mark=seq_y_mark[:, -self.pred_len:, :]
        )

        return out


def criterion(args, model):
    return nn.MSELoss()
