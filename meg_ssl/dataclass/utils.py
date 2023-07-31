import numpy as np


def z_score_epoch(epoch:np.ndarray, baseline_mean:np.ndarray=None, baseline_std:np.ndarray=None)->np.ndarray:
    """
    epoch: ch x time
    """
    if baseline_mean is None:
        baseline_mean = np.mean(epoch, axis=1) # 時間方向の平均
    if baseline_std is None:
        baseline_std = np.std(epoch, axis=1) # 時間方向の分散
    return (epoch - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]

def car_epoch(epoch:np.ndarray)->np.ndarray:
    """
    epoch: ch x time
    """
    return epoch - np.mean(epoch, axis=0)

def clamp_epoch(epoch, a_min, a_max):
    """
    epoch: ch x time
    """
    return np.clip(epoch, a_min, a_max)