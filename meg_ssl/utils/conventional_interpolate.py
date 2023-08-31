import torch
import numpy as np

def linear_interpolate(data:torch.Tensor, mask:torch.Tensor):
    # data: (n, L, c)
    # mask: (n, L)
    raise NotImplementedError
    pred_data = data.copy()
    pred_data[mask,:] = -1


def polynomial_interpolate(data:np.ndarray, mask:np.ndarray, patch_size, order=3):
    # data: (n, c, L)
    # mask: (n, L)
    pred_data  = data.copy()
    unpatched_mask_batch = []
    for n in range(len(data)):
        unpatched_mask = np.zeros((len(mask[n])*patch_size))
        for i, s in enumerate(np.arange(0, len(mask[n])*patch_size, patch_size)):
            unpatched_mask[s:s+patch_size] = mask[n,i]
        unpatched_mask_batch.append(unpatched_mask)
        for c in range(data.shape[1]):
            masked_x = np.where(unpatched_mask==1)[0]
            masked_data = data[n,c,:][masked_x]

            unmasked_x = np.where(unpatched_mask==0)[0]
            unmasked_data = data[n,c,:][unmasked_x]
            # import pdb; pdb.set_trace()
            if len(unmasked_data) == 0:
                continue
            p = np.polyfit(unmasked_x, unmasked_data, order)

            pred = np.polyval(p, masked_x)
            # import pdb; pdb.set_trace()
            pred_data[n, c, masked_x] = pred
    # unpatched_mask_batch = np.stack(unpatched_mask_batch, axis=0)
    return pred_data # , unpatched_mask_batch


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
    gt: [N, num_electrodes, num_samples]
    gt: [N, chan, T]
    pred: [N, num_electrodes, num_samples] 
    mask: [N, num_samples], 0 is keep, 1 is remove,
    """
    unpatched_mask_batch = []
    for n in range(len(gt)):
        unpatched_mask = np.zeros((len(mask[n])*patch_size))
        for i, s in enumerate(np.arange(0, len(mask[n])*patch_size, patch_size)):
            unpatched_mask[s:s+patch_size] = mask[n,i]
        unpatched_mask_batch.append(unpatched_mask)
    mask = np.stack(unpatched_mask_batch, axis=0)

    # import pdb; pdb.set_trace()
    # gt = gt.transpose(1,2)
    target = gt
    # target = patchify(gt, patch_size)
    # target = gt.transpose(1,2)
    loss = (pred - target) ** 2
    # import pdb; pdb.set_trace()
    loss = loss.mean(axis=1)  # [N, L], mean loss per patch
    # loss = loss.mean()
    loss = (loss * mask).sum() / mask.sum()  if mask.sum() != 0 else (loss * mask).sum() # mean loss on removed patches

    corr =  np.mean(np.array([np.corrcoef(np.stack([p[0], s[0]],axis=0))[0,1] for p, s in zip(pred, gt)]))
    return loss, corr

