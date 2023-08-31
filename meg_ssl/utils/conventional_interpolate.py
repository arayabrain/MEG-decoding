import torch
import numpy as np

def linear_interpolate(data:torch.Tensor, mask:torch.Tensor):
    # data: (n, L, c)
    # mask: (n, L)
    raise NotImplementedError
    pred_data = data.copy()
    pred_data[mask,:] = -1


def polynomial_interpolate(data:np.ndarray, mask:np.ndarray, order=3):
    # data: (n, L, c)
    # mask: (n, L)
    pred_data  = data.copy()
    for n in range(len(data)):
        for c in range(data.shape[-1]):
            masked_data = data[n,:,c][mask[n,:]]
            if len(masked_data) == 0:
                continue
            x = np.where(mask[n,:]==True)[0]
            p = np.polyfit(x, masked_data, order)

            masked_x = np.where(mask[n,:]==False)[0]
            gt = data[n,:,c][masked_x]
            pred = np.polyval(p, masked_x)
            pred_data[n, masked_data, c] = pred

    return pred_data


def patchify(imgs, patch_size):
    """
    imgs: (N, 1, num_voxels)
    imgs: [N, chan, T]
    x: (N, L, patch_size)
    x: [N, chan * 4, T/4]
    """
    p = patch_size
    assert imgs.ndim == 3 and imgs.shape[1] % p == 0, 'ndim: {}, chan: {}, p: {}'.format(imgs.ndim, imgs.shape[1], p)

    # h = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1] // p, -1))
    return x

def unpatchify(x, patch_size):
    """
    x: (N, L, patch_size)
    imgs: (N, 1, num_voxels)
    """
    p = patch_size
    h = x.shape[1]

    imgs = x.reshape(shape=(x.shape[0], -1, x.shape[2] // p))
    return np.transpose(imgs,(1,2))

def calc_loss_and_corr(gt:np.ndarray, pred:np.ndarray, mask:np.ndarray, patch_size:int=4):
    """
    gt: [N, 1, num_voxels]
    gt: [N, chan, T]
    pred: [N, L, p]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    gt = gt.transpose(1,2)
    target = patchify(gt, patch_size)
    # target = gt.transpose(1,2)
    loss = (pred - target) ** 2
    loss = loss.mean(axis=-1)  # [N, L], mean loss per patch
    # loss = loss.mean()
    loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches

    corr =  np.mean(np.tensor([np.corrcoef(np.stack([p[0], s[0]],axis=0))[0,1] for p, s in zip(pred, gt)]))
    return loss, corr

