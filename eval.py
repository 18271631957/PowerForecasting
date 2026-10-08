import datetime
import json
from types import SimpleNamespace

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

from dataloader.data_loader import get_data_loader
from main import get_model
from utils.logger import get_logger
import pandas as pd
import matplotlib.pyplot as plt

from utils.processing_result import post_process

# 2. 定义需要执行的训练任务列表（核心配置组合）
eval_tasks = [
    # {"model_name": "TimeXer", "csv_path": "./dataset/DKASC/DKASC_ready.csv",
    #  "checkpoint_path": "TimeXer_20260624_231606",
    #  "train_range": ("2020-01-01", "2021-12-31"),
    #  "val_range": ("2022-01-01", "2022-12-31"), "test_range": ("2023-01-01", "2023-12-31"),
    #  "date_col_name": "timestamp", "cap": 23.4,
    #  "target": "Active_Power", "features": "MS", "patch_len": 16, "n_heads": 8, "e_layers": 1,
    #  "enc_in": 2, "d_model": 256, "dropout": 0.05, "d_ff": 256, "activation": "gelu", "scale": True,
    #  "RevIn": True, "get_pred_type": "last"},

    # {
    #     "model_name": "TimeXer",
    #     "checkpoint_path": "TimeXer_20260922_234754",
    #     "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #     "train_range": ("2007-01-01", "2015-12-31"),
    #     "val_range": ("2015-12-31", "2017-12-31"),
    #     "test_range": ("2017-12-31", "2019-12-31"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "timestamp",
    #     "target": "active_power",
    #     "features": "MS", "patch_len": 16, "n_heads": 8, "e_layers": 1, "enc_in": 2, "d_model": 256, "dropout": 0.05,
    #     "d_ff": 256, "activation": "gelu", "RevIn": True,
    # },

    # {
    #     "checkpoint_path": "TimeXer_20260926_175649",
    #     "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #     "test_range": ("2017-12-31", "2019-12-31"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    # },

    # {
    #     "checkpoint_path": "TimeXer_20260928_224126",
    #     "csv_path": "./dataset/YinDian/Yindian_20180716_20230329_align_clean.csv", "cap": 100,
    #     "test_range": ("2022-01-30", "2022-04-30"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    # },

    {
        "checkpoint_path": "PMDformerMS_Gansu_20260930_235511",
        # "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
        # "test_range": ("2017-12-31", "2019-12-31"),
        "task": "short_term",
        "sample_stride": 96,
    },
]
# 3. 遍历执行每个训练任务
if __name__ == "__main__":
    for idx, task in enumerate(eval_tasks):
        print(f"\n========== 执行第 {idx + 1}/{len(eval_tasks)} 个训练任务 ==========")

        # 定位模型的文件夹路径
        model_name = task["checkpoint_path"].split("_")[0]
        # model_name = "PMDformerMS_BSAPGF"
        checkpoint_dir = Path("./checkpoints") / model_name / task["checkpoint_path"]
        config_path = checkpoint_dir / "config.json"

        # 读取模型训练时的配置
        with open(config_path, 'r', encoding='utf-8') as f:
            saved_args_dict = json.load(f)
            saved_args_dict['is_train'] = False  # 设置为eval模式
            saved_args_dict['plot_flag'] = True  # 要画图

        # 根据task更新训练时的参数，因为可能想eval不同的测试集
        saved_args_dict.update(task)
        args = SimpleNamespace(**saved_args_dict)  # 转回SimpleNamespace对象，让后面的代码依然可以用 args.xxx 访问

        # 设置测试结果的保存路径
        test_path = checkpoint_dir / f"eval"
        test_path.mkdir(parents=True, exist_ok=True)
        checkpoint_abs_path = test_path.parent / "weights" / "checkpoint_best.pth"

        # 获取和数据集
        test_data, test_loader = get_data_loader(args, flag='test')
        # 将数据集自动识别出的纯净 NWP 索引直接赋给 args / configs
        args.valid_nwp_indices = test_data.valid_nwp_indices

        # 根据参数创建模型并加载权重
        model, criterion = get_model(args)
        model = model.to(args.device)
        model.load_state_dict(torch.load(checkpoint_abs_path, map_location=args.device), strict=False)

        # 创建当前测试目录
        cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
        save_root = test_path / f"{args.model_name}_{cur_time}" / "result"
        save_root.mkdir(parents=True, exist_ok=True)
        # 获取日志
        log_eval = get_logger(save_root.parent, 'eval')

        # 测试
        pred_list = []
        true_list = []
        test_loss_list = []

        model.eval()  # 将模型设置为评估模式，关闭 Dropout 和 BatchNorm 的更新
        with torch.no_grad():  # 不计算梯度，节省显存并加速
            for i, (seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future) in tqdm(enumerate(test_loader),
                                                                              total=len(test_loader),
                                                                              desc=f"TEST"):
                seq_x = seq_x.float().to(args.device)
                seq_y = seq_y.float().to(args.device)
                seq_x_mark = seq_x_mark.float().to(args.device)
                seq_y_mark = seq_y_mark.float().to(args.device)
                seq_future = seq_future.float().to(args.device)
                output = model(seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future)  # 模型预测
                loss = criterion(output, seq_y)  # 计算损失
                test_loss_list.append(loss.item())

                # 反标准化拿真实功率
                for _i in range(output.shape[0]):  # 遍历batch_size
                    # 预测值反标准化 - 整个序列(16,1)
                    pred_seq = output[_i].cpu().detach().numpy()  # 形状: (16, 1)
                    if args.scale:
                        pred_inverse = test_data.inverse_transform(pred_seq)  # 反标准化整个序列
                    else:
                        pred_inverse = pred_seq
                    # 真实值反标准化 - 整个序列(pred_len,1)
                    true_seq = seq_y[_i].cpu().detach().numpy()  # 形状: (16, 1)
                    if args.scale:
                        true_inverse = test_data.inverse_transform(true_seq)  # 反标准化整个序列
                    else:
                        true_inverse = true_seq
                    pred_all = pred_inverse.flatten()  # 展平为一维数组
                    true_all = true_inverse.flatten()
                    pred_list.append(pred_all)
                    true_list.append(true_all)
                    # pred_list.extend(pred_all.tolist())
                    # true_list.extend(true_all.tolist())

        post_process(pred_list, save_root, args)
        test_loss_avg = np.average(test_loss_list)
        log_eval.info(f"Avg Loss: {test_loss_avg}")
