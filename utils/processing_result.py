import logging
import shutil
from pathlib import Path

from models.MachineLearning.XGBoost.DataSetUtils import split_time_dataset
from utils.Metrics import calculate_mse, calculate_mae, calculate_rmse, calculate_nse, calculate_mape, NW_ACC4H, \
    GW_ACC4H
import pandas as pd
import matplotlib.dates as mdates
import os
import matplotlib.pyplot as plt
import numpy as np


def post_process(pred_list, test_folder_path, args):
    if args.task == 'ultra_short_term':
        post_process_ultra_short_term(pred_list, test_folder_path, args)
    elif args.task == 'short_term':
        post_process_short_term(pred_list, test_folder_path, args)
    else:
        raise ValueError("task 只能填ultra_short_term或者short_term.")


def post_process_ultra_short_term(pred_list, test_folder_path, args):
    time_col_name = args.date_col_name
    target_col_name = args.target
    # 保存预测结果
    _, val_df, df_test = split_time_dataset(
        csv_path=args.csv_path,
        train_range=args.train_range,
        val_range=args.val_range,
        test_range=args.test_range
    )
    df_test[time_col_name] = pd.to_datetime(df_test[time_col_name])

    full_pred = [None] * args.seq_len + pred_list + (args.pred_len - 1) * [None]  # 创建带None填充的完整预测列
    print(f"测试集长度:{len(df_test)}|预测数组长度:{len(full_pred)}")
    df_test['pred_list'] = full_pred
    df_test.to_csv(Path(test_folder_path) / f"{args.model_name}_pred_all.csv", index=False)  # 保存文件

    # 按需求保存某个时间步的csv
    df_test['pred'] = df_test['pred_list'].apply(lambda x: x[-1] if x is not None else None).shift(args.pred_len)

    # -------------------------------------后处理-------------------------------------
    df_test['pred'] = df_test['pred'].where(df_test['pred'] >= 0, 0)  # 负值设置0
    hours = df_test[time_col_name].dt.hour
    mask = (hours < 6) | (hours >= 20)
    df_test.loc[mask, 'pred'] = 0  # 早上5点 晚上8点后设置0
    df_test.to_csv(Path(test_folder_path) / f"{args.model_name}_{args.get_pred_type}_pred.csv", index=False)
    # -------------------------------------后处理完毕-------------------------------------

    print(f"正在对每一天的真实功率vs预测，并计算每天的指标...")
    figures_path = Path(test_folder_path) / "figures"
    os.makedirs(figures_path, exist_ok=True)
    daily_results = []  # 用于存储每天的指标
    for date, group in df_test.groupby(df_test[time_col_name].dt.date):
        # 绘制真实值和预测值
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(group[time_col_name], group[target_col_name], label='True', color='blue')
        ax.plot(group[time_col_name], group['pred'], label='Pred', color='red', linestyle='--')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置横轴格式：只显示 时:分
        ax.set_title(f"PV Power Prediction - {date}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(figures_path / f"{date}.png", bbox_inches='tight')  # 保存并关闭（防止内存溢出）
        plt.close()

        # 计算
        true_vals = group[target_col_name].values
        pred_vals = group['pred'].values
        mse = calculate_mse(true_vals, pred_vals)
        mae = calculate_mae(true_vals, pred_vals)
        rmse = calculate_rmse(true_vals, pred_vals)
        nse = calculate_nse(true_vals, pred_vals)
        mape = calculate_mape(true_vals, pred_vals)
        nw_acc = NW_ACC4H(true_vals, pred_vals, Cap=int(args.cap))
        gw_acc = GW_ACC4H(true_vals, pred_vals, Cap=int(args.cap))

        # 将指标存入列表
        daily_results.append({
            'Date': date,
            'MSE': mse, 'MAE': mae, 'RMSE': rmse,
            'NSE': nse, 'MAPE': mape,
            'NW_ACC': nw_acc, 'GW_ACC': gw_acc
        })

    # 3. 保存每日指标统计表
    metrics_df = pd.DataFrame(daily_results)
    metrics_file = Path(test_folder_path) / "daily_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False, float_format='%.4f')
    print(f"绘图完成！每日指标已保存至: {metrics_file}")

    # ===================== 你要新增的逻辑 =====================
    log_test = logging.getLogger('test')
    df = metrics_df.iloc[1:].copy()
    mean_df = df.drop(columns=['Date']).mean().to_frame().T
    log_test.info("\n%s", mean_df.round(4).to_string(index=False))


def post_process_short_term(pred_list, test_folder_path, args, zenith_col_name='zenith'):
    """
    :param pred_list: 预测结果列表。每一个item都是96个时间步（一整天）的数据
    """
    time_col_name = args.date_col_name
    target_col_name = args.target

    df_real = pd.read_csv(args.csv_path)
    df_real[time_col_name] = pd.to_datetime(df_real[time_col_name])

    # 1. 截取测试集时间范围的数据
    start_dt = pd.to_datetime(args.test_range[0])
    end_dt = pd.to_datetime(args.test_range[1]).replace(hour=23, minute=59, second=59)
    df_test = df_real[(df_real[time_col_name] >= start_dt) & (df_real[time_col_name] <= end_dt)]

    # 2. 提取测试集中实际存在的所有日期（精确到天，去重）
    # .dt.floor('D') 会把 "2022-01-03 14:15:00" 变成 "2022-01-03 00:00:00"
    all_existing_days = df_test[time_col_name].dt.floor('D').drop_duplicates().tolist()
    existing_days_set = set(all_existing_days)

    # 3. 筛选有效预测日：只有当“昨天”和“前天”的数据都存在时，“今天”才能被预测
    days = []
    for cur_day in all_existing_days:
        yesterday = cur_day - pd.Timedelta(days=1)
        two_days_ago = cur_day - pd.Timedelta(days=2)

        if (yesterday in existing_days_set) and (two_days_ago in existing_days_set):
            days.append(cur_day)

    if len(pred_list) != len(days):
        raise ValueError(f"数据维度不匹配！预测结果有 {len(pred_list)} 天，但有效连续日期有 {len(days)} 天。")

    records = []
    # 按天循环（遍历天数索引）
    for i, day in enumerate(days):
        cur_issue_time = day - pd.Timedelta(days=1) + pd.Timedelta(hours=8, minutes=45)  # 早上8点45发布预报
        day_target_times = pd.date_range(start=day, periods=96, freq="15min")  # 当天96个时间点
        day_preds = pred_list[i]  # 当天的96个预测值
        for t_time, power in zip(day_target_times, day_preds):
            records.append({
                "target_time": t_time,
                "issue_time": cur_issue_time,
                "pred": power,
            })
    df = pd.DataFrame(records)

    df_real = pd.read_csv(args.csv_path)
    df_real[time_col_name] = pd.to_datetime(df_real[time_col_name])
    # 使用左连接 (left join) 根据 target_time 拼接 real_power 列
    df = pd.merge(df,
                  df_real[[time_col_name, target_col_name, zenith_col_name]],
                  left_on="target_time",  # 左表 df 里的时间列
                  right_on=time_col_name,  # 右表 df_real 里的时间列
                  how='left')
    # df.drop(columns=[time_col_name], inplace=True)  # 删除右表带入的多余时间列
    issue_file = Path(test_folder_path) / "issue_file.csv"
    df.to_csv(issue_file, index=False)

    # -------------------------------------后处理-------------------------------------
    # 下限负值截断为 0，上限截断为装机容量 Cap（防超调）
    df['pred'] = df['pred'].clip(lower=0.0, upper=float(args.cap))
    # 天顶角夜间截断（阈值设为 89° 比较稳妥，你也可以观察数据微调为 88° 或 90°）
    mask = df['zenith'] >= 88.0
    df.loc[mask, 'pred'] = 0.0
    df.to_csv(issue_file.with_name("issue_file_postpre.csv"), index=False)
    # -------------------------------------后处理完毕-------------------------------------

    print(f"正在对每一天的真实功率vs预测，并计算每天的指标...")
    figures_path = Path(test_folder_path) / "figures"
    os.makedirs(figures_path, exist_ok=True)
    daily_results = []  # 用于存储每天的指标
    for date, group in df.groupby(df[time_col_name].dt.date):
        # 绘制真实值和预测值
        if args.plot_flag:
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(group[time_col_name], group[target_col_name], label='True', color='blue')
            ax.plot(group[time_col_name], group['pred'], label='Pred', color='red', linestyle='--')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 设置横轴格式：只显示 时:分
            ax.set_title(f"PV Power Prediction - {date}")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.savefig(figures_path / f"{date}.png", bbox_inches='tight')  # 保存并关闭（防止内存溢出）
            plt.close()

        # 计算
        true_vals = group[target_col_name].values
        pred_vals = group['pred'].values
        mse = calculate_mse(true_vals, pred_vals)
        mae = calculate_mae(true_vals, pred_vals)
        rmse = calculate_rmse(true_vals, pred_vals)
        nse = calculate_nse(true_vals, pred_vals)
        mape = calculate_mape(true_vals, pred_vals)
        nw_acc = NW_ACC4H(true_vals, pred_vals, Cap=int(args.cap))
        gw_acc = GW_ACC4H(true_vals, pred_vals, Cap=int(args.cap))

        # 将指标存入列表
        daily_results.append({
            'Date': date,
            'MSE': mse, 'MAE': mae, 'RMSE': rmse,
            'NSE': nse, 'MAPE': mape,
            'NW_ACC': nw_acc, 'GW_ACC': gw_acc
        })

    # 3. 保存每日指标统计表
    metrics_df = pd.DataFrame(daily_results)
    metrics_file = Path(test_folder_path) / "daily_metrics.csv"
    metrics_df.to_csv(metrics_file, index=False, float_format='%.4f')
    print(f"绘图完成！每日指标已保存至: {metrics_file}")

    # ===================== 你要新增的逻辑 =====================
    if args.is_train:
        logger = logging.getLogger('test')
    else:
        logger = logging.getLogger('eval')
    df = metrics_df.iloc[1:].copy()
    mean_df = df.drop(columns=['Date']).mean().to_frame().T
    logger.info("\n%s", mean_df.round(4).to_string(index=False))
