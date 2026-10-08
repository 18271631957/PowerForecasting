import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    适合可学习嵌入的场景：
        数据量充足（避免过拟合）
        时间模式复杂（多层周期性、非线性关系）
        领域特定（不同行业的时间模式差异很大）
        长期预测（需要捕获复杂的长期依赖）
    """

    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        if embed_type == 'fixed':
            embed = FixedEmbedding
        else:
            embed = nn.Embedding
        if freq == 't':
            self.minute_embed = embed(4, d_model)
        """
            hour_embed[24, d_model]0-23小时的嵌入向量
            weekday_embed[7, d_model]0-6星期的嵌入向量
            day_embed[32, d_model]1-31日期的嵌入向量
            month_embed[13, d_model]1-12月份的嵌入向量
            
            x = torch.tensor([
                [[3, 15, 2, 14],    # 样本1-时间步1: 3月15日周二14:00
                 [3, 16, 3, 9],     # 样本1-时间步2: 3月16日周三09:00  
                 [3, 17, 4, 16]],   # 样本1-时间步3: 3月17日周四16:00
                
                [[4, 1, 5, 8],      # 样本2-时间步1: 4月1日周五08:00
                 [4, 2, 6, 12],     # 样本2-时间步2: 4月2日周六12:00
                 [4, 3, 0, 20]]     # 样本2-时间步3: 4月3日周日20:00
            ])
            根据数据表示，所以hour_embed和weekday_embed刚好是0-23，0-6. 但是day_embed和month_embed的表示范围是1-31和1-12
            nn.Embedding(32,64) 也就是Embedding会创建行为0-31，列为0-63的参数矩阵，但是我们使用的时候只适应1-31行
            x[:, :, 1]为形状：[2, 3]
            [
                [15, 16, 17],   
                [1, 2, 3]
            ]
            x[:, :, 1] - 1为
            [
                [14, 15, 16],   
                [0, 1, 2]
            ]
        """
        self.hour_embed = embed(24, d_model)
        self.weekday_embed = embed(7, d_model)
        self.day_embed = embed(32, d_model)
        self.month_embed = embed(13, d_model)

    def forward(self, x):
        x = x.long()
        # 如果 freq='t'（分钟级别），才会创建 minute_embed 属性
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])
        """
            经过训练后，时间嵌入向量会自动学习到：
                周期性：星期一的向量和下一个星期一的向量相似
                相关性：相邻时间的向量更相似（如14:00和15:00）
                语义性：工作时间和休息时间在向量空间中分离
            这样，原本离散、稀疏的时间信息就被转换成了稠密、连续、语义丰富的向量表示，为下游的神经网络提供了强大的时间感知能力！
        """
        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    # timeF的嵌入方式
    def __init__(self, d_model, freq='h'):
        super(TimeFeatureEmbedding, self).__init__()
        # 'h': 小时 → 4个特征
        # 't': 分钟 → 5个特征
        # 's': 秒 → 6个特征
        # 'm': 月 → 1个特征
        # 'a': 年 → 1个特征
        # 'w': 周 → 2个特征
        # 'd': 日 → 3个特征
        # 'b': 工作日 → 3个特征
        freq_map = {"15min": 5, 'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # (batch_size, seq_len, n_vars) -> (batch_size, n_vars, seq_len)
        x = x.permute(0, 2, 1)
        if x_mark is None:
            # 直接进行值嵌入 (batch_size, n_vars, seq_len) -> (batch_size, d_model, seq_len)
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_wo_temp(nn.Module):
    # wo_temp无时间嵌入
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_temp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
        LLMMixer/Informer
    """

    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class DataEmbedding_wo_pos_decoder(nn.Module):
    # wo_pos_decoder无位置编码的解码器专用版本
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_decoder, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        if embed_type == 'timeF':
            self.temporal_embedding = TimeFeatureEmbedding(d_model=d_model, freq=freq)
        else:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        x_mark = self.temporal_embedding(x_mark)
        x_mark_expanded = x_mark.repeat(1, 7, 1)
        x = x + x_mark_expanded
        return self.dropout(x)


class DataEmbedding_wo_pos_temp(nn.Module):
    # wo是without的简写.wo_pos_temp也就是没有pos位置编码和temp时间嵌入
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos_temp, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)
