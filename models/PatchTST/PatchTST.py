import torch
from torch import nn
import math
from .layers.Transformer_EncDec import Encoder, EncoderLayer
from .layers.SelfAttention_Family import FullAttention, AttentionLayer

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        # 一维数据填充的层，通过复制输入张量的边界值来扩展数据长度，常用于卷积神经网络中保持输出尺寸或保护边界信息
        # (0, padding)代表 左侧不填充（填充量为0）右侧填充padding个元素
        # 若输入为[A, B, C]，padding=2时输出为[A, B, C, C, C]（左侧不填充，右侧复制两次C）
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        # 将序列映射到高维
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # 位置编码
        self.position_embedding = PositionalEmbedding(d_model)
        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        # 填充边缘
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class CrossChannelFusionHead(nn.Module):
    """
    跨通道融合头：__init__不再传入n_vars，n_vars运行时从输入shape获取
    input:  [bs, n_vars, patch_num, d_model]
    output: [bs, pred_len, 1]
    """
    def __init__(self, patch_num, d_model, pred_len, dropout=0.):
        super().__init__()
        self.patch_num = patch_num
        self.d_model = d_model
        self.pred_len = pred_len

        # 通道维度聚合：将任意n_vars个通道向量聚合为1个全局向量
        self.channel_agg = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 可学习的通道权重（投影，不绑定n_vars）
        self.channel_attn_proj = nn.Linear(d_model, 1)

        self.out_mlp = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model//2, pred_len)
        )

    def forward(self, x):
        # x: [B, n_vars, patch_num, d_model]
        B, n_vars, patch_num, d_model = x.shape

        # 1. 在patch维度做平均池化，每个通道得到1个全局表征 [B, n_vars, d_model]
        x_channel = x.mean(dim=2)

        # 2. 对每个通道做变换
        x_channel = self.channel_agg(x_channel) # [B, n_vars, d_model]

        # 3. 计算每个通道的重要性权重(不绑定n_vars数量)
        attn_score = self.channel_attn_proj(x_channel)  # [B, n_vars, 1]
        attn_weight = torch.softmax(attn_score, dim=1)  # 沿通道做softmax

        # 4. 加权求和聚合所有通道信息得到全局向量 [B, d_model]
        global_feat = torch.sum(x_channel * attn_weight, dim=1)

        # 5. 映射输出预测长度
        out = self.out_mlp(global_feat) # [B, pred_len]
        return out.unsqueeze(-1)         # [B, pred_len, 1]


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(configs.d_model, patch_len, stride, padding, configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(
                Transpose(1, 2),
                nn.BatchNorm1d(configs.d_model),
                Transpose(1, 2))
        )

        # Prediction Head
        patch_count = int((configs.seq_len - patch_len) / stride + 2)
        self.head_nf = configs.d_model * patch_count
        # self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        # 替换原来FlattenHead为跨通道融合头
        self.head = CrossChannelFusionHead(
            n_vars=configs.enc_in,
            patch_num=patch_count,
            d_model=configs.d_model,
            pred_len=self.pred_len,
            dropout=configs.dropout
        )

    def forward(self, seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future, mask=None):

        # Normalization from Non-stationary Transformer
        means = seq_x.mean(1, keepdim=True).detach()
        seq_x = seq_x - means
        stdev = torch.sqrt(
            torch.var(seq_x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        seq_x /= stdev

        # do patching and embedding
        seq_x = seq_x.permute(0, 2, 1)
        # u: [bs * nvars, patch_num, d_model]
        enc_out, n_vars = self.patch_embedding(seq_x)

        # Encoder
        # z: [bs * nvars, patch_num, d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs, nvars, patch_num, d_model]
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))

        # Decoder 映射成时间序列
        dec_out = self.head(enc_out)  # [B, pred_len, 1] 单变量输出

        # --------反归一化注意：我们预测的是目标列（功率，原最后一列），用目标列的统计量恢复---------
        # means/stdev: [B,1,N]，取目标变量维度（假设目标是最后一维）
        target_idx = -2
        dec_out = dec_out * stdev[:, 0:1, target_idx:target_idx + 1]
        dec_out = dec_out + means[:, 0:1, target_idx:target_idx + 1]

        return dec_out[:, -self.pred_len:, :]  # [B, L, D]

def criterion(args, model):
    return nn.MSELoss()
