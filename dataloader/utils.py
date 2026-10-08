import pandas as pd


def build_lag_features(df_raw, target_col='ACTIVEPOWER'):
    """
    为原始 DataFrame 批量构造滞后特征
    注意：输入 df 必须按时间顺序排好，且包含对应列
    """
    df = df_raw.copy()
    # 1. 批量构造 1-96 阶深层滞后 (目标值与辐射量)
    cols_96 = [target_col, 'Radiation_Global_Tilted']
    lags_96 = pd.concat([df[cols_96].shift(i).add_suffix(f'_lag_{i}') for i in range(1, 97)], axis=1)

    # 2. 批量构造 1 阶辅助特征滞后
    aux_cols = ['Current_Phase_Average',
                'Weather_Temperature_Celsius',
                'Weather_Relative_Humidity',
                'Global_Horizontal_Radiation',
                'Diffuse_Horizontal_Radiation',
                'Wind_Direction',
                'Weather_Daily_Rainfall']
    lags_1 = df[aux_cols].shift(1).add_suffix('_lag_1')

    # 3. 拼接并抛弃前 96 行包含 NaN 的数据 (重置索引防止 DataLoader 报错)
    df_featured = pd.concat([df, lags_96, lags_1], axis=1).dropna().reset_index(drop=True)

    return df_featured