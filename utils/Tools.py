import numpy as np
import torch
import logging

from utils.logger import get_logger


# 让JSON属性可以通过.来访问, 让代码看起来更整洁
class DotDict(dict):
    """一个简单的类，使得字典属性可以通过点号访问"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    # lr_adjust = {epoch: learning_rate * (0.8 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0, save_mode=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.save_mode = save_mode
        self.solar_logger = logging.getLogger('val')

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.solar_logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            self.solar_logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        # 2. 创建一个只包含“可训练”参数的 state_dict
        trainable_state_dict = {name: param for name, param in model.named_parameters() if param.requires_grad}
        # 3. 保存这个小得多的 state_dict
        torch.save(trainable_state_dict, path / 'checkpoint_best.pth')

        self.val_loss_min = val_loss