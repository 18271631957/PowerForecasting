import os
import shutil

from models.DLinear import DLinear
from models.LSTM_Transformer import LSTM_Transformer
from models.TimeXer import TimeXer
from models.PatchTST import PatchTST
from models.PMDformerMS import PMDformerMS
from models.PMDformerMSBSAPGF import PMDformerMSBSAPGF
from models.PMDformer import PMDformer
from models.MyLinear import MyLinear

from utils.Tools import EarlyStopping
from utils.processing_result import post_process

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime
import random
import time
from pathlib import Path
import numpy as np
import torch
from utils.logger import get_logger
from dataloader.data_loader import get_data_loader
from mmengine.optim.scheduler.lr_scheduler import PolyLR
from tqdm import tqdm
import json


def log_model_param(logger, model):
    trained_parameters = []
    numel_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)
            num_params = p.numel()
            numel_params = numel_params + num_params
            logger.info(f"Layer: {name}, Parameters: {num_params}")
    logger.info(f"All Parameters Num: {len(trained_parameters)}, All Numel:{numel_params}")
    return trained_parameters


def get_model(args):
    model_dict = {
        "DLinear": DLinear,
        "TimeXer": TimeXer,
        "PatchTST": PatchTST,
        "PMDformer": PMDformer,
        "PMDformerMS": PMDformerMS,
        "PMDformerMSBSAPGF": PMDformerMSBSAPGF,
        "LSTM_Transformer": LSTM_Transformer,
        "MyLinear": MyLinear,
    }
    model_cls = model_dict[args.model_name]
    model = model_cls.Model(args)
    criterion = model_cls.criterion(args, model)  # 统一传参，内部自己决定用不用 model
    return model, criterion


def main(args):
    checkpoints_path = "./checkpoints"
    cur_time = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    dataset_name = os.path.basename(os.path.dirname(args.csv_path))  # YinDian
    process_folder_path = os.path.join(checkpoints_path, args.model_name,
                                       args.model_name + '_' + dataset_name + '_' + cur_time)
    Path(process_folder_path).mkdir(parents=True, exist_ok=True)

    config_path = os.path.join(process_folder_path, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)  # vars(args) 会把你的 SimpleNamespace 对象转换成普通字典，方便 json 保存

    log_train = get_logger(process_folder_path, 'train')
    log_val = get_logger(process_folder_path, 'val')
    log_test = get_logger(process_folder_path, 'test')

    log_train.info("args -> " + str(args))
    args.device = torch.device(args.device)


    # 获取数据
    train_data, train_loader = get_data_loader(args, flag='train')
    vali_data, vali_loader = get_data_loader(args, flag='val')
    test_data, test_loader = get_data_loader(args, flag='test')
    # 将数据集自动识别出的纯净 NWP 索引直接赋给 args / configs
    args.valid_nwp_indices = train_data.valid_nwp_indices

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 创建模型
    model, criterion = get_model(args)
    # 打印参数
    log_model_param(log_train, model)
    model.to(args.device)



    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    log_train.info(f'The number of sample {len(train_loader)} {len(vali_loader)} {len(test_loader)}')
    params = model.parameters()

    # 模型优化器
    if args.sgd:
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # 学习率迭代器
    if args.lr_scheduler == 'StepLR':
        # 阶梯式学习率：每经过 args.lr_drop 个 epoch，学习率乘以 gamma（默认0.1）
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    elif args.lr_scheduler == 'CosAWS':
        # T_0=30    第一个重启周期的长度（step/epoch）
        # T_mult=2  每个后续周期长度为上一周期的2倍（周期逐渐变长）
        # eta_min=1e-5  学习率下降的最小值，不会低于该值
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-5)
    elif args.lr_scheduler == 'PolyLR':
        # 多项式衰减学习率：学习率平滑缓慢下降至最小值
        # eta_min=args.min_lr  学习率最低值
        # begin=args.start_epoch  开始衰减的epoch
        # end=args.epochs  衰减结束的epoch（总训练轮数）
        lr_scheduler = PolyLR(optimizer, eta_min=args.min_lr, begin=args.start_epoch, end=args.epochs)
    elif args.lr_scheduler == 'ReduceLR':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20,
                                                                  verbose=True)
    elif args.lr_scheduler == 'CosLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    elif args.lr_scheduler == 'OneCycleLR':
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, steps_per_epoch=len(train_loader), pct_start=0.2,
                                                           div_factor=10, epochs=args.epochs, max_lr=args.learning_rate)
    else:
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    output_dir = Path(process_folder_path) / 'weights'
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    all_metrics_history = []  # 累计指标
    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        criterion.train()

        # train
        # 记录每次预测模型的损失
        train_loss_list = []
        epoch_time = time.time()  # epoch开始时间
        for i, (seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future) in tqdm(enumerate(train_loader),
                                                                          total=len(train_loader),
                                                                          desc=f"TRAIN Epoch {epoch}"):
            optimizer.zero_grad()
            seq_x = seq_x.float().to(args.device)
            seq_y = seq_y.float().to(args.device)
            seq_x_mark = seq_x_mark.float().to(args.device)
            seq_y_mark = seq_y_mark.float().to(args.device)
            seq_future = seq_future.float().to(args.device)
            output = model(seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future)  # 模型预测
            # print(output)
            loss = criterion(output, seq_y)  # 计算损失

            loss.backward()  # 反向传播
            optimizer.step()  # 根据梯度更新模型参数
            train_loss_list.append(loss.item())

        train_loss_avg = np.average(train_loss_list)  # 训练集、验证集和测试集损失（当前epoch平均）
        cur_lr = optimizer.param_groups[0]['lr']
        log_train.info(f"Epoch: {epoch} | Avg Loss: {train_loss_avg} | LR: {cur_lr:}")
        if args.lr_scheduler == 'ReduceLR':
            lr_scheduler.step(train_loss_avg)  # ReduceLROnPlateau 需要传入监控指标（这里用训练平均损失）
        else:
            lr_scheduler.step()  # 其他调度器（StepLR/CosLR/PolyLR）直接step

        # 验证
        with torch.no_grad():
            model.eval()
            criterion.eval()

            # 验证
            val_loss_list = []
            for i, (seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future) in tqdm(enumerate(vali_loader),
                                                                              total=len(vali_loader),
                                                                              desc=f"VAL Epoch {epoch}"):
                seq_x = seq_x.float().to(args.device)
                seq_y = seq_y.float().to(args.device)
                seq_x_mark = seq_x_mark.float().to(args.device)
                seq_y_mark = seq_y_mark.float().to(args.device)
                seq_future = seq_future.float().to(args.device)
                output = model(seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future)  # 模型预测
                loss = criterion(output, seq_y)  # 计算损失
                val_loss_list.append(loss.item())

            val_loss_avg = np.average(val_loss_list)
            log_val.info(f"Epoch: {epoch} | Avg Loss: {val_loss_avg}")
            # 早停
            early_stopping(val_loss_avg, model, output_dir)

            # 测试
            if early_stopping.counter == 0:  # 有新的best
                pred_list = []
                true_list = []
                test_loss_list = []
                for i, (seq_x, seq_x_mark, seq_y, seq_y_mark, seq_future) in tqdm(enumerate(test_loader),
                                                                                  total=len(test_loader),
                                                                                  desc=f"TEST Epoch {epoch}"):
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
                        # 真实值反标准化 - 整个序列(16,1)
                        true_seq = seq_y[_i].cpu().detach().numpy()  # 形状: (16, 1)
                        if args.scale:
                            true_inverse = test_data.inverse_transform(true_seq)  # 反标准化整个序列
                        else:
                            true_inverse = true_seq
                        pred_all = pred_inverse.flatten()  # 展平为一维数组
                        true_all = true_inverse.flatten()  # 展平为一维数组
                        pred_list.append(pred_all)
                        true_list.append(true_all)
                        # pred_list.extend(pred_all.tolist())
                        # true_list.extend(true_all.tolist())

                test_folder_path = output_dir.parent / "test" / f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_test_{epoch}'
                test_folder_path.mkdir(parents=True, exist_ok=True)

                post_process(pred_list, test_folder_path, args)
                test_loss_avg = np.average(test_loss_list)
                log_test.info(f"Epoch: {epoch} | Avg Loss: {test_loss_avg}")

        if early_stopping.early_stop:
            log_val.info("Early stopping")
            break

    # 计算整个过程花了多少时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    log_train.info('Process time {}'.format(total_time_str))
