import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    """位置编码，为Transformer提供序列位置信息"""

    # 使Transformer能够感知序列中元素的相对位置

    # 构造函数，d_model为模型维度，max_len为最大序列长度（默认5000）
    def __init__(self, d_model, max_len=5000):
        # 调用父类构造函数
        super().__init__()
        # 初始化位置编码矩阵，形状为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # 创建位置索引向量，形状为(max_len, 1)，表示0到max_len-1的位置
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 计算位置编码的除数项，使用指数衰减的三角函数频率
        # 公式中的10000^(2i/d_model)部分，取对数后计算
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        # 对偶数索引位置（0, 2, 4...）使用正弦函数编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数索引位置（1, 3, 5...）使用余弦函数编码
        pe[:, 1::2] = torch.cos(position * div_term)
        # 调整维度为(max_len, 1, d_model)，增加batch维度便于广播
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将位置编码注册为buffer（不作为模型参数训练，但会随模型保存）
        self.register_buffer('pe', pe)

    # 前向传播函数
    def forward(self, x):
        # x输入形状: (seq_len, batch_size, d_model)
        # 将位置编码加到输入x上，使用广播机制自动扩展
        x = x + self.pe[:x.size(0), :]
        # 返回添加位置信息后的张量
        return x


class Model(nn.Module):
    # 构造函数，定义模型各层参数
    # def __init__(self, input_dim=17, seq_len=96, d_model=128,
    #              lstm_hidden=128, lstm_layers=2,
    #              nhead=8, num_transformer_layers=2,
    #              dim_feedforward=256, dropout=0.1):
    def __init__(self, args, input_dim=17, seq_len=96, d_model=128,
                 lstm_hidden=128, lstm_layers=2,
                 nhead=8, num_transformer_layers=2,
                 dim_feedforward=256, dropout=0.1):
        # 调用父类构造函数
        super().__init__()

        # 保存输入维度（特征数17）
        self.input_dim = input_dim
        # 保存序列长度（96个时间步）
        self.seq_len = seq_len
        # 保存模型维度（Transformer的d_model）
        self.d_model = d_model

        # 输入投影层：将17维特征映射到d_model维度（128维）
        # 这是必要的，因为Transformer需要固定维度的输入
        self.input_projection = nn.Linear(input_dim, d_model)

        # 实例化位置编码模块
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)

        # LSTM编码器：提取时序特征
        # batch_first=False表示输入格式为(seq_len, batch, feature)
        # dropout仅在lstm_layers>1时生效，防止层数少时警告
        self.lstm = nn.LSTM(d_model, lstm_hidden, lstm_layers,
                            batch_first=False, dropout=dropout if lstm_layers > 1 else 0)

        # LSTM到Transformer的投影层：如果LSTM隐藏层维度与d_model不同，需要线性变换
        # 如果相同，则使用恒等映射（不做变换）
        self.lstm_to_transformer = nn.Linear(lstm_hidden, d_model) if lstm_hidden != d_model else nn.Identity()

        # 定义Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # 模型维度128
            nhead=nhead,  # 多头注意力头数8
            dim_feedforward=dim_feedforward,  # 前馈网络维度256
            dropout=dropout,  # dropout率0.1
            batch_first=False  # 使用(seq_len, batch, feature)格式
        )
        # 堆叠多个Transformer编码器层（默认2层）
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 定义输出层，使用Sequential容器组织多层网络
        # 从d_model维度映射到64维，再映射到96维（预测96个时间点的功率值）
        # self.output_layer = nn.Sequential(
        #     nn.Linear(d_model, 64),  # 第一层线性变换：128->64
        #     nn.ReLU(),  # ReLU激活函数引入非线性
        #     nn.Dropout(dropout),  # Dropout正则化防止过拟合
        #     nn.Linear(64, 96)  # 第二层线性变换：64->96，输出96个预测值
        # )
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),  # 第一层线性变换：128->64
            nn.ReLU(),  # ReLU激活函数引入非线性
            nn.Dropout(dropout),  # Dropout正则化防止过拟合
            nn.Linear(64, 96)  # 第二层线性变换：64->96，输出96个预测值
        )

        # Dropout层，用于前向传播中的随机失活
        self.dropout = nn.Dropout(dropout)

    # 前向传播函数，定义数据流向
    # def forward(self, x):
    def forward(self, x, seq_x_mark, seq_y, seq_y_mark, seq_future):
        # x输入形状: (batch, seq_len, input_dim) = (N, 96, 17)
        # 获取批次大小
        batch_size = x.size(0)

        # 1. 输入投影: 将17维特征映射到d_model维度
        # 形状变化: (N, 96, 17) -> (N, 96, 128)
        x = self.input_projection(x)

        # 2. 调整维度为(seq_len, batch, d_model)以适应LSTM和Transformer
        # 形状变化: (N, 96, 128) -> (96, N, 128)
        x = x.transpose(0, 1)

        # 3. 添加位置编码，使模型感知序列位置信息
        # 形状保持: (96, N, 128)
        x = self.pos_encoder(x)
        # 应用dropout正则化
        x = self.dropout(x)

        # 4. LSTM编码: 提取时序特征
        # lstm_out形状: (96, N, lstm_hidden)，hidden和cell为最后一个时间步的隐藏状态
        lstm_out, (hidden, cell) = self.lstm(x)

        # 5. 维度映射: 如果LSTM输出维度与Transformer输入维度不同，进行线性变换
        # 形状变化: (96, N, lstm_hidden) -> (96, N, 128)
        lstm_out = self.lstm_to_transformer(lstm_out)

        # 6. Transformer编码: 通过自注意力机制捕获全局依赖关系
        # 形状保持: (96, N, 128)
        transformer_out = self.transformer_encoder(lstm_out)

        # 7. 取最后一个时间步的Transformer输出作为全局表示
        # 形状变化: (96, N, 128) -> (N, 128)
        last_output = transformer_out[-1, :, :]

        # 8. 输出层: 将128维特征映射到96个预测值（对应96个时间点的功率）
        # 形状变化: (N, 128) -> (N, 96)
        output = self.output_layer(last_output)
        output = output.unsqueeze(-1)
        # 返回预测结果
        return output


def criterion(args, model):
    return nn.MSELoss()
