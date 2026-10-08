import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from models.MachineLearning.XGBoost.DataSetUtils import split_time_dataset
from .timefeatures import time_features


class Day_ahead_Pv_15minute(Dataset):
    def __init__(self,
                 df_raw,
                 run_flag,
                 size=[96 + 36, 96],  # [历史长度, 预测长度] 前一天96+当天36（00：00-08：45）
                 date_col_name="OBSERVETIME",
                 target='ACTIVEPOWER',
                 zenith_col='zenith',
                 scale=True,
                 scalers=None,
                 sample_stride=1,
                 blind_spot_len=60):
        self.df_raw = df_raw.reset_index(drop=True)  # 重置 DataFrame 的行索引，并直接丢弃旧索引，让数据重新获得一个从 0 开始的干净、连续的数字索引。
        self.run_flag = run_flag
        self.seq_len, self.pred_len = size
        self.target = target
        self.scale = scale
        self.date_col_name = date_col_name
        self.zenith_col = zenith_col
        self.scalers = scalers
        self.sample_stride = sample_stride
        self.blind_spot_len = blind_spot_len  # 盲区长度，默认60步 (09:00至23:45)

        if run_flag in ['test', 'val'] and not self.scalers:
            raise ValueError("训练集/验证集需要 训练集的Scalers!")

        self._read_data()

        # 校验物理时间连续性
        self.raw_timestamps = pd.to_datetime(self.df_raw[self.date_col_name])
        # 总跨度 = 历史 + 盲区 + 未来
        total_steps = self.seq_len + self.blind_spot_len + self.pred_len
        freq_delta = pd.Timedelta(minutes=15)
        expected_diff = freq_delta * (total_steps - 1)
        shifted_time = self.raw_timestamps.shift(-(total_steps - 1))
        valid_mask = (shifted_time - self.raw_timestamps) == expected_diff

        all_valid_starts = np.where(valid_mask)[0]
        if self.sample_stride == 1:
            self.valid_indices = all_valid_starts.tolist()
        else:
            # 严格基于时间间隔对齐起点
            start_times = self.raw_timestamps.iloc[all_valid_starts]
            base_time = start_times.iloc[0]
            stride_delta = freq_delta * self.sample_stride
            time_diffs = start_times - base_time
            aligned_mask = (time_diffs % stride_delta) == pd.Timedelta(0)
            self.valid_indices = all_valid_starts[aligned_mask].tolist()

        self.tot_len = len(self.valid_indices)

        original_len = (len(self.data_x) - total_steps) // self.sample_stride + 1
        print(f"[{self.run_flag} set] 理论最大样本: {original_len} | 剔除断层有效样本: {self.tot_len}")

    def _read_data(self):
        df_data = self.df_raw.drop(columns=[self.date_col_name])
        if self.target not in df_data.columns:
            raise ValueError(f"目标变量 '{self.target}' 不在数据列中。")
        if self.zenith_col not in df_data.columns:
            raise ValueError(f"目标变量 '{self.zenith_col}' 不在数据列中。")

        target_data = df_data.pop(self.target)
        zenith_data = df_data.pop(self.zenith_col)
        # 目标列放在到第二列  天顶角放在倒第一列
        df_data[self.target] = target_data
        df_data[self.zenith_col] = zenith_data
        # 找到目标列的索引
        self.target_idx = df_data.columns.get_loc(self.target)
        # 动态找出所有需要防泄露的列索引（目标列 + 所有以'SK'开头的实况气象列）
        self.leakage_indices = []
        self.valid_nwp_indices = []
        for index, col_name in enumerate(df_data.columns):
            is_target = (col_name == self.target)
            is_sk_feature = str(col_name).startswith('SK')
            is_zenith = (col_name == self.zenith_col)

            if is_target or is_sk_feature:
                self.leakage_indices.append(index)
            elif not is_zenith:
                self.valid_nwp_indices.append(index)

        print(df_data.columns)
        features = df_data.values
        targets = df_data[[self.target]].values

        if self.scale:
            if self.run_flag == 'train':
                self.feature_scaler = StandardScaler().fit(features)
                self.target_scaler = StandardScaler().fit(targets)
            else:
                self.feature_scaler = self.scalers['feature']
                self.target_scaler = self.scalers['target']

            self.data_x = self.feature_scaler.transform(features)
            self.data_y = self.target_scaler.transform(targets)
        else:
            self.data_x = features
            self.data_y = targets

        timestamps = pd.to_datetime(self.df_raw[self.date_col_name].values)
        self.data_stamp = time_features(timestamps, freq='15min').transpose(1, 0)

    def __getitem__(self, index):
        actual_start = self.valid_indices[index]
        # 1. 历史观测阶段
        x_end = actual_start + self.seq_len

        # 2. 跨越时间盲区 (跳过 09:00 - 23:45)
        y_begin = x_end + self.blind_spot_len

        # 3. 预测未来阶段
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[actual_start:x_end]
        seq_x_mark = self.data_stamp[actual_start:x_end]

        seq_y = self.data_y[y_begin:y_end]
        seq_y_mark = self.data_stamp[y_begin:y_end]

        # 获取未来的 NWP 与天文特征，并强制将最后一列（实际功率）归零防止泄露
        seq_future = self.data_x[y_begin:y_end].copy()
        # seq_future[:, self.target_idx] = 0.0
        seq_future[:, self.leakage_indices] = 0.0
        # print(seq_future[:, -2])
        return seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future

    def inverse_transform(self, data):
        return self.target_scaler.inverse_transform(data)

    def __len__(self):
        return self.tot_len


def get_data_loader(args, flag, shuffle_flag=True, drop_last=False):
    assert flag in ['train', 'val', 'test']
    shuffle_flag = shuffle_flag if flag == 'train' else False

    train_df, val_df, test_df = split_time_dataset(
        csv_path=args.csv_path,
        train_range=args.train_range,
        val_range=args.val_range,
        test_range=args.test_range,
        time_col=args.date_col_name
    )

    train_dataset = Day_ahead_Pv_15minute(
        df_raw=train_df, run_flag='train',
        target=args.target, date_col_name=args.date_col_name, scale=True, sample_stride=args.sample_stride
    )

    if flag == 'train':
        data_set = train_dataset
    else:
        target_df = val_df if flag == 'val' else test_df
        scalers = {'feature': train_dataset.feature_scaler, 'target': train_dataset.target_scaler}
        data_set = Day_ahead_Pv_15minute(
            df_raw=target_df, run_flag=flag, size=[args.seq_len, args.pred_len],
            target=args.target, date_col_name=args.date_col_name,
            scale=True, scalers=scalers, sample_stride=args.sample_stride
        )

    data_loader = DataLoader(
        data_set, batch_size=args.batch_size, shuffle=shuffle_flag,
        num_workers=args.num_workers, drop_last=drop_last
    )

    return data_set, data_loader
