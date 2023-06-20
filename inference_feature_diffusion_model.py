from meg_decoding.diffusion_utils.models import UNet
from meg_decoding.diffusion_utils.helpers import get, get_default_device, GODDenoiseDataset, get_dataloader, setup_log_directory
from meg_decoding.diffusion_utils.diffusion_process import SimpleDiffusion, forward_diffusion
from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from torch.cuda import amp
from torchmetrics import MeanMetric
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import gc


@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = "god_train" #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("/home/yainoue/meg2image/results/dp", "logs")
    root_checkpoint_dir = os.path.join("/home/yainoue/meg2image/results/dp", "checkpoints")
    root_output_dir = '/home/yainoue/meg2image/results/20230429_sbj01_eegnet_cv_norm_regression/features/dp'

    # Current log and checkpoint directory.
    checkpoint_dir = "version_7"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    INPUT_SHAPE = (1, 512)
    NUM_EPOCHS = 800
    BATCH_SIZE = 32
    LR = 2e-4
    NUM_WORKERS = 2

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8
    APPLY_ATTENTION = (False, True, True, False)#(True, True, True, True)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128

def get_dataset(dataset_name='god_train', slice_=None):
    if 'god' in dataset_name:
        split = dataset_name.split('_')[1]
        dataset_path = '/home/yainoue/meg2image/results/20230429_sbj01_eegnet_cv_norm_regression/features/{kind}_{split}.npy'

        dataset = GODDenoiseDataset(dataset_path.format(kind='z', split=split), 
                                    dataset_path.format(kind='y', split=split), 
                                    dataset_path.format(kind='l', split=split),
                                    slice_=slice_)
    return dataset

def diffusion_and_denoise(device, checkpoint_dir, slice_=None):
    model = UNet(
        input_channels          = TrainingConfig.INPUT_SHAPE[0],
        output_channels         = TrainingConfig.INPUT_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])
    model.to(BaseConfig.DEVICE)
    model.eval()

    dataset = get_dataset(dataset_name=BaseConfig.DATASET, slice_=slice_)

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.INPUT_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    z_embeddings = []
    xts_embeddings = []
    y_embeddings = []
    l_list = []
    ts = torch.ones(1, dtype=torch.long, device=BaseConfig.DEVICE) * (TrainingConfig.TIMESTEPS-1)
    
    for z,y,l in tqdm(dataset):
        z = z.to(device)
        y = y.to(device)
        z_embeddings.append(z)
        x0s = y# z
        xts, _ = forward_diffusion(sd, x0s, ts)

        n_noise = torch.randn_like(xts)
        predicted_noise = model(xts, ts)
        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        xts = (
            one_by_sqrt_alpha_t
            * (xts - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * n_noise
        )

        xts = xts.detach().squeeze(0)
        y = y.detach()
        xts_embeddings.append(xts)
        y_embeddings.append(y)
        l_list.append(l)
        gc.collect()
    xts_embeddings = torch.cat(xts_embeddings, dim=0)
    y_embeddings = torch.cat(y_embeddings, dim=0)
    z_embeddings = torch.cat(z_embeddings, dim=0)
    l_list = np.array(l_list)
    return z_embeddings, xts_embeddings, y_embeddings, l_list

def calculate_correlation(embedding:torch.Tensor, savedir:str, savename:str):
    corr_coeff = torch.corrcoef(embedding)
    plt.figure(figsize=(10,10))
    plt.imshow(corr_coeff.detach().cpu().numpy(), vmin=-0.5, vmax=0.5)
    plt.colorbar()
    plt.savefig(os.path.join(savedir, savename))
    print('save to ', savedir, savename)
    plt.close()

def run():

    savedir = os.path.join(BaseConfig.root_output_dir, BaseConfig.DATASET)
    os.makedirs(savedir, exist_ok=True)
    z_embeddings, xts_embeddings, y_embeddings, l_list = diffusion_and_denoise(BaseConfig.DEVICE, 
                                                                               os.path.join(BaseConfig.root_checkpoint_dir, 
                                                                                            BaseConfig.checkpoint_dir),
                                                                                slice_=slice(0,600,1))
    print(l_list)
    calculate_correlation(z_embeddings, savedir, 'z_corr.png')
    calculate_correlation(xts_embeddings, savedir, 'xts_corr.png')
    calculate_correlation(y_embeddings, savedir, 'y_corr.png')
    calculate_correlation(torch.cat([z_embeddings, y_embeddings]), savedir, 'z_y_corr.png')
    calculate_correlation(torch.cat([xts_embeddings, y_embeddings]), savedir, 'z_xts_corr.png')


if __name__ == '__main__':
    run()
