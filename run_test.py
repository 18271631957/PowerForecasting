from types import SimpleNamespace
from main import main


# 1. 定义默认基础参数（复用原有默认值）
def get_default_args():
    # 数据集相关基础参数（dataset_path后续会被覆盖）
    data_args = SimpleNamespace(
        scale=True,
        batch_size=32,
        num_workers=1,
    )

    task_args = SimpleNamespace(
        seq_len=96 + 36,
        pred_len=96,
        n_input_features=1,
        get_pred_type='all'
    )
    # 优化器/学习率参数
    optim_args = SimpleNamespace(
        lr_scheduler='PolyLR',
        lr=5e-4,
        min_lr=1e-6,
        weight_decay=0.01,
        epochs=50,
        patience=15,
        start_epoch=0,
        lr_drop=30,
        sgd=False,
    )

    # 通用基础参数（model_name后续会被覆盖）
    common_args = SimpleNamespace(
        device='cuda',
        seed=42,
        model_name='XXX',
        sava_all_checkpoint=False,
        is_train=True,
        plot_flag=False,
    )
    # 合并所有默认参数
    args = SimpleNamespace(
        **vars(task_args),
        **vars(data_args),
        **vars(optim_args),
        **vars(common_args)
    )
    return args


train_tasks = [
    # {
    #     "model_name": "TimeXer",
    #     # "csv_path": "./dataset/DKASC/DKASC_20140101_20231231_2_3.csv", "cap": 23.4,
    #     "csv_path": "./dataset/DKASC/DKASC_20140101_20231231_2_3_4.csv", "cap": 23.4,
    #     "train_range": ("2014-01-01", "2021-02-17"),
    #     "val_range": ("2021-01-01", "2021-12-31"),
    #     "test_range": ("2022-06-30", "2022-12-31"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "timestamp",
    #     "target": "Active_Power", "features": "MS", "patch_len": 16, "n_heads": 8, "e_layers": 1,
    #     "enc_in": 2, "d_model": 256, "dropout": 0.05, "d_ff": 256, "activation": "gelu",
    #     "RevIn": True,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {"model_name": "LSTM_Transformer",
    #  "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #  "train_range": ("2007-01-01", "2015-12-31"),
    #  "val_range": ("2015-12-31", "2017-12-31"),
    #  "test_range": ("2017-12-31", "2019-12-31"),
    #  "task": "short_term",
    #  "sample_stride": 96,
    #  "target": "active_power",
    #  "date_col_name": "timestamp",
    #  "lr": 0.001,
    #  "lr_scheduler": "ReduceLR",
    #  "epochs": 500,
    #  "patience": 50,
    #  },

    # 当TimeXer使用天顶角zenith当内生变量时，准确率居然更高
    # {
    #     "model_name": "TimeXer",
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
    #     "epochs": 500,
    #     "patience": 50,
    # },
    # #
    # {
    #     "model_name": "TimeXer",
    #     "csv_path": "./dataset/YinDian/Yindian_20180716_20230329_align_clean.csv", "cap": 100,
    #     "train_range": ("2018-07-16", "2020-12-31"),
    #     "val_range": ("2020-12-31", "2021-12-31"),
    #     "test_range": ("2022-07-01", "2022-09-30"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "OBSERVETIME",
    #     "target": "ACTIVEPOWER",
    #     "features": "MS", "patch_len": 16, "n_heads": 8, "e_layers": 1, "enc_in": 2, "d_model": 256, "dropout": 0.05,
    #     "d_ff": 256, "activation": "gelu", "RevIn": True,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {
    #     "model_name": "PatchTST",
    #     "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #     "train_range": ("2007-01-01", "2015-12-31"),
    #     "val_range": ("2015-12-31", "2017-12-31"),
    #     "test_range": ("2017-12-31", "2019-12-31"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "timestamp",
    #     "target": "active_power",
    #     "patch_len": 16, "stride": 8, "n_heads": 8, "e_layers": 2, "enc_in": 17, "d_model": 512, "dropout": 0.05,
    #     "d_ff": 2048, "activation": "gelu",
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {
    #     "model_name": "PMDformerMS",
    #     "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #     "train_range": ("2007-01-01", "2015-12-31"),
    #     "val_range": ("2015-12-31", "2017-12-31"),
    #     "test_range": ("2017-12-31", "2019-12-31"),
    #     "date_col_name": "timestamp",
    #     "target": "active_power",
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "e_layers": 2, "v_layers": 1, "n_heads": 8, "patch_size": 16, "d_model": 128, "d_ff": 128, "RevIn": True,
    #     "dropout": 0.1, "d_future": 17, "d_mark": 5,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {
    #     "model_name": "PMDformer",
    #     "csv_path": "./dataset/Gansu/Gansu_2007_2020_15min_solar.csv", "cap": 150,
    #     "train_range": ("2007-01-01", "2015-12-31"),
    #     "val_range": ("2015-12-31", "2017-12-31"),
    #     "test_range": ("2017-12-31", "2019-12-31"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "timestamp",
    #     "target": "active_power",
    #     "e_layers": 2, "v_layers": 1, "n_heads": 8, "patch_size": 16, "d_model": 128, "d_ff": 128, "RevIn": True,
    #     "dropout": 0.1,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {
    #     "model_name": "PMDformer",
    #     "csv_path": "./dataset/YinDian/Yindian_20180716_20230329_align_clean.csv", "cap": 100,
    #     "train_range": ("2018-07-16", "2020-12-31"),
    #     "val_range": ("2020-12-31", "2021-12-31"),
    #     "test_range": ("2022-07-01", "2022-09-30"),
    #     "date_col_name": "OBSERVETIME",
    #     "target": "ACTIVEPOWER",
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "e_layers": 2, "v_layers": 1, "n_heads": 8, "patch_size": 16, "d_model": 128, "d_ff": 128, "RevIn": True,
    #     "dropout": 0.1,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    # {
    #     "model_name": "PMDformerMS",
    #     "csv_path": "./dataset/YinDian/Yindian_20180716_20230329_align_clean.csv", "cap": 100,
    #     "train_range": ("2018-07-16", "2020-12-31"),
    #     "val_range": ("2020-12-31", "2021-12-31"),
    #     "test_range": ("2022-07-01", "2022-09-30"),
    #     "task": "short_term",
    #     "sample_stride": 96,
    #     "date_col_name": "OBSERVETIME",
    #     "target": "ACTIVEPOWER",
    #     "e_layers": 2, "v_layers": 1, "n_heads": 8, "patch_size": 16, "d_model": 128, "d_ff": 128, "RevIn": True,
    #     "dropout": 0.1, "d_future": 11, "d_mark": 5,
    #     "epochs": 500,
    #     "patience": 50,
    # },

    {
        "model_name": "PMDformerMS_BSAPGF",
        "csv_path": "./dataset/YinDian/Yindian_20180716_20230329_align_clean_delAll0.csv", "cap": 100,
        "train_range": ("2018-07-16", "2020-12-31"),
        "val_range": ("2020-12-31", "2021-12-31"),
        "test_range": ("2021-12-31", "2022-12-31"),
        "task": "short_term",
        "sample_stride": 96,
        "date_col_name": "OBSERVETIME",
        "target": "ACTIVEPOWER",
        "e_layers": 2, "v_layers": 1, "n_heads": 8, "patch_size": 16, "d_model": 128, "d_ff": 128, "RevIn": True,
        "dropout": 0.1, "d_future": 11, "d_mark": 5,
        "epochs": 500,
        "patience": 50,
    },


]
# 3. 遍历执行每个训练任务c
if __name__ == "__main__":
    for idx, task in enumerate(train_tasks):
        args = get_default_args()
        for key, value in task.items():
            setattr(args, key, value)

        # 打印任务信息（便于日志追踪）
        print(f"\n========== 执行第 {idx + 1}/{len(train_tasks)} 个训练任务 ==========")
        print(f"模型名称: {args.model_name}")
        print(f"数据集路径: {args.csv_path}")
        print(f"调度器: {args.lr_scheduler} | 学习率: {args.lr}")  # 顺便打印出来确认一下
        print("=" * 50)

        main(args)
