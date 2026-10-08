import torch
import torch.nn as nn


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 计算需要的填充长度 并在前端填充和后端填充
        padding_length = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, padding_length, 1)
        end = x[:, -1:, :].repeat(1, padding_length, 1)
        x = torch.cat([front, x, end], dim=1)
        # (batch_size,seq_len,n_vars) -> (batch_size,n_vars,seq_len)
        x = x.permute(0, 2, 1)
        # 假如(64,2,96) 2为辐射量，功率。2和96合起来理解就是，要对辐射量、功率两个特征都进行池化。
        x = self.avg(x)
        # (batch_size,n_vars,seq_len) -> (batch_size,seq_len,n_vars)
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)  # 趋势Trend(t)
        res = x - moving_mean  # 季节性成分Seasonal(t)
        return res, moving_mean


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.decompsition = series_decomp(configs.moving_avg)
        self.individual = configs.individual
        self.channels = configs.n_input_features

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):  # 多变量
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))


        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, seq_x_mark, seq_y, seq_y_mark, seq_future ):
        x = x[:, :, -1:]  # 取功率
        seasonal_init, trend_init = self.decompsition(x)
        # (batch_size,seq_len,n_vars) -> (batch_size,n_vars,seq_len)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            # 创建预测结果矩阵
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            # 对多变量进行预测
            for i in range(self.channels):
                # 对第i个变量进行预测 self.Linear_Seasonal[i]表示第i个变量的季节线性层权重
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            # (batch_size,n_vars,seq_len) -> (batch_size,n_vars,pred_len)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)  # to [Batch, Output length, Channel]


def criterion(args, model):
    return nn.MSELoss()
