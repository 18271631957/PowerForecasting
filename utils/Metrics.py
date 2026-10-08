import numpy as np


def calculate_mse(true, pred):
    """均方误差 Mean Squared Error"""
    return np.mean((true - pred) ** 2)


def calculate_mae(true, pred):
    """平均绝对误差 Mean Absolute Error"""
    return np.mean(np.abs(true - pred))


def calculate_rmse(true, pred):
    """均方根误差 Root Mean Squared Error"""
    return np.sqrt(calculate_mse(true, pred))


# def calculate_mape(true, pred):
#     """
#     平均绝对百分比误差 Mean Absolute Percentage Error
#     注意：光伏数据中含有0，直接除以0会报错。
#     通常做法：只计算真实值大于某一阈值（如0.1）的点，或者加一个极小值 epsilon。
#     这里采用加 epsilon 的方式防止除零。
#     """
#     epsilon = 1e-6
#     return np.mean(np.abs((true - pred) / (true + epsilon))) * 100

def calculate_mape(true, pred, threshold=0.01):
    """
    平均绝对百分比误差 Mean Absolute Percentage Error
    注意：光伏数据中含有0，直接除以0会报错。
    通常做法：只计算真实值大于某一阈值（如0.1）的点，或者加一个极小值 epsilon。
    这里采用加 epsilon 的方式防止除零。
    """
    # 生成有效掩码，只保留真实值大于阈值的样本
    valid_mask = true > threshold
    y_true_valid = true[valid_mask]
    y_pred_valid = pred[valid_mask]

    # 防止有效样本为空时报错
    if len(y_true_valid) == 0:
        return np.nan

    epsilon = 1e-6
    mape = np.mean(np.abs((y_true_valid - y_pred_valid) / (y_true_valid + epsilon))) * 100
    return mape


def calculate_nse(true, pred):
    """
    纳什效率系数 Nash-Sutcliffe Efficiency
    NSE = 1 - ( sum((obs - pred)^2) / sum((obs - mean_obs)^2) )
    """
    numerator = np.sum((true - pred) ** 2)
    denominator = np.sum((true - np.mean(true)) ** 2)
    if denominator == 0:
        return float('-inf')  # 方差为0，无法计算
    return 1 - (numerator / denominator)


def NW_ACC4H(true_list, pred_list, Cap):
    """
    计算功率预测的准确率。
    【已修改】本版本支持任意长度的输入列表（例如每日80个点）。
    """
    n_points = len(true_list)

    # 【修改点 2】: 放宽对列表长度的严格限制
    # 只要列表不为空且长度匹配即可
    if n_points == 0 or n_points != len(pred_list):
        return None
    sum_squares = 0.0
    for i in range(n_points):
        ti = true_list[i]
        pi = pred_list[i]
        if ti >= 0.2 * Cap:
            denominator = ti
        else:
            denominator = 0.2 * Cap
        # 防止除以零的错误
        if denominator == 0:
            # 如果分母为0，且分子也为0（预测和实际都为0），则误差为0
            if ti - pi == 0:
                error_term = 0
            # 如果分母为0，但分子不为0，这是一个较大的误差，可以设为1（或其它惩罚值）
            else:
                error_term = 1
        else:
            error_term = (ti - pi) / denominator
        sum_squares += error_term ** 2

    # 【修改点 4】: 除数使用实际数据点数 n_points
    rmse = (sum_squares / n_points) ** 0.5
    accuracy = (1 - rmse) * 100

    return accuracy





def GW_ACC4H(true_list, pred_list, Cap):
    """
    国网标准 (通常与上述公式类似，或是双细则积分电量算法)
    此处暂定与 NW_ACC4H 逻辑一致 (1-RMSE/Cap)，如有特定公式可在此修改。
    """
    # 如果您的 GW 标准是基于 '准确率 = 1 - 绝对误差/Cap' (即MAE based)，请修改此处
    # 目前保持与 SCI 论文通用的 RMSE-based 准确率一致
    rmse = calculate_rmse(true_list, pred_list)
    return (1 - (rmse / Cap)) * 100

# def NW_ACC4H(true_list, pred_list, Cap):
#     """
#     西北监能局/图片公式: Acc = (1 - RMSE/Cap) * 100%
#     """
#     rmse = calculate_rmse(true_list, pred_list)
#     if Cap == 0: return 0
#     return (1 - (rmse / Cap)) * 100