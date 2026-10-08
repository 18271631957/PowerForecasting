import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers.Transformer_Enc import Encoder_TRA, EncoderLayer_TRA, Encoder_PVA, EncoderLayer_PVA
from .layers.SelfAttention_Family import TrendRestorationAttention, SelfAttention

from einops import rearrange

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        # self.enc_in = configs.enc_in
        self.e_layers = configs.e_layers
        self.RevIn = configs.RevIn
        self.patch_size = configs.patch_size
        self.stride = configs.patch_size
        
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        
        # Embdding Layer and Prediction Layer
        self.Embedding_layer = nn.Linear(self.patch_size, configs.d_model)
        self.Predicting_layer = nn.Linear(configs.d_model * (self.patch_num), configs.pred_len)

        self.Encoder_TRA = Encoder_TRA(
            [
                EncoderLayer_TRA(
                        TrendRestorationAttention(d_model=configs.d_model, n_heads=configs.n_heads, dropout=configs.dropout),
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

    def forward(self, seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future, mask=None):
        x = seq_x
        B, L, N = x.shape 
        
        x = rearrange(x, 'b l n -> b n l')
        if self.RevIn:
            means = x.mean(-1, keepdim=True)
            stdev = torch.sqrt(torch.var(x, dim=-1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - means) / stdev

        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  #[B N PI PS]
        x = rearrange(x, 'b n p s -> (b n) p s')    #[B*N PI PS]
        
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
        
        x = self.Predicting_layer(x)    #[B N T]

        if self.RevIn:
            x = x * stdev + means

        x = rearrange(x, 'b n t -> b t n')

        # return x
        # return x[:, :, -2:-1]
        return x[:, :, -1:]

def criterion(args, model):
    return nn.MSELoss()
