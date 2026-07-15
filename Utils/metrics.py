import numpy as np


def compute_rmse(y_true, y_pred):
    """
    计算 Root Mean Square Error (RMSE)。

    参数：
      y_true: 真实值，numpy 数组
      y_pred: 预测值，numpy 数组

    返回：
      RMSE 值 (float)
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def compute_mae(y_true, y_pred):
    """
    计算 Mean Absolute Error (MAE)。

    参数：
      y_true: 真实值，numpy 数组
      y_pred: 预测值，numpy 数组

    返回：
      MAE 值 (float)
    """
    return np.mean(np.abs(y_true - y_pred))


def compute_pearson(y_true, y_pred):
    """
    计算 Pearson's correlation coefficient (R)。

    参数：
      y_true: 真实值，numpy 数组（1D）
      y_pred: 预测值，numpy 数组（1D）

    返回：
      Pearson 相关系数 (float)，取值范围在 [-1, 1]
    """
    # 保证是一维数组
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    if y_true.size == 0:
        return np.nan
    corr_matrix = np.corrcoef(y_true, y_pred)
    return corr_matrix[0, 1]


def compute_sd(y_true, y_pred):
    """
    计算预测误差的标准差 (Standard Deviation, SD)。
    这里定义 SD 为预测误差（y_true - y_pred）的标准差。

    参数：
      y_true: 真实值，numpy 数组
      y_pred: 预测值，numpy 数组

    返回：
      标准差值 (float)
    """
    errors = y_true - y_pred
    return np.std(errors)


def evaluate_metrics(y_true, y_pred):
    """
    同时计算所有指标，并返回一个字典，包含 RMSE、MAE、Pearson (R) 和 SD。

    参数：
      y_true: 真实值，numpy 数组
      y_pred: 预测值，numpy 数组

    返回：
      metrics: 字典，键包括 "RMSE", "MAE", "Pearson", "SD"
    """
    metrics = {
        "RMSE": compute_rmse(y_true, y_pred),
        "MAE": compute_mae(y_true, y_pred),
        "Pearson": compute_pearson(y_true, y_pred),
        "SD": compute_sd(y_true, y_pred)
    }
    return metrics



