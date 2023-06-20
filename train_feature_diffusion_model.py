from meg_decoding.diffusion_utils.models import UNet
from meg_decoding.diffusion_utils.helpers import get, get_default_device, GODFeatureDataset, get_dataloader, setup_log_directory
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
    DATASET = ["god_train", "coco"] #  "MNIST", "Cifar-10", "Cifar-100", "Flowers"

    # For logging inferece images and saving checkpoints.
    root_log_dir = os.path.join("/home/yainoue/meg2image/results/dp", "logs")
    root_checkpoint_dir = os.path.join("/home/yainoue/meg2image/results/dp", "checkpoints")

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"

@dataclass
class TrainingConfig:
    TIMESTEPS = 1000 # Define number of diffusion timesteps
    INPUT_SHAPE = (1, 512)
    NUM_EPOCHS = 800
    BATCH_SIZE = 64
    LR = 2e-5# 2e-4
    NUM_WORKERS = 2

@dataclass
class ModelConfig:
    BASE_CH = 64  # 64, 128, 256, 256
    BASE_CH_MULT = (1, 2, 4, 4) # 32, 16, 8, 8
    APPLY_ATTENTION = (False, True, True, False)#(True, True, True, True)
    DROPOUT_RATE = 0.1
    TIME_EMB_MULT = 4 # 128

def get_dataset(dataset_names=['god_train']):
    dataset_list = []
    for dataset_name in dataset_names:
        if 'god' in dataset_name:
            split = dataset_name.split('_')[1]
            dataset_path = f'/home/yainoue/meg2image/results/20230429_sbj01_eegnet_cv_norm_regression/features/y_{split}.npy'
            slice_ = slice(0,1200,1) if split == 'train' else None
            dataset = GODFeatureDataset(dataset_path, slice=slice_)
        elif 'coco' in dataset_name:
            dataset_path = f'/home/yainoue/meg2image/codes/MEG-decoding/data/COCO/unlabeled2017_features.pkl'
            slice_=None
            dataset = GODFeatureDataset(dataset_path, slice=slice_, dim=0)
        else:
            pass
        dataset_list.append(dataset)
    dataset = torch.utils.data.ConcatDataset(dataset_list)
    print('dataset_length: ', len(dataset))
    return dataset

def difussion_demo(savedir, device):
    sd = SimpleDiffusion(num_diffusion_timesteps=TrainingConfig.TIMESTEPS, device="cuda")
    dataset = get_dataset(dataset_names=BaseConfig.DATASET)
    loader = iter(      # converting dataloader into an iterator for now.
                get_dataloader(
                    dataset=dataset,
                    batch_size=1,
                    device="cuda",
                )
    )
    x0s = next(loader)
    x0s = x0s.to(device)
    noisy_images = []
    specific_timesteps = [0, 10, 50, 100, 150, 200, 250, 300, 400, 600, 800, 999]

    for timestep in specific_timesteps:
        timestep = torch.as_tensor(timestep, dtype=torch.long).to(device)

        xts, _ = forward_diffusion(sd, x0s, timestep)

        noisy_images.append(xts)

    # Plot and see samples at different timesteps

    _, ax = plt.subplots(1, len(noisy_images), figsize=(10, 5), facecolor="white")

    for i, (timestep, noisy_sample) in enumerate(zip(specific_timesteps, noisy_images)):
        noisy_sample = noisy_sample.squeeze()
        noisy_sample = noisy_sample.detach().cpu().numpy()
        ax[i].plot(np.arange(len(noisy_sample)), noisy_sample)
        ax[i].set_title(f"t={timestep}", fontsize=8)
        ax[i].axis("off")
        ax[i].grid(False)

    plt.suptitle("Forward Diffusion Process", y=0.9)
    plt.axis("off")
    savefile = os.path.join(savedir, "forward_diffusion.png")
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()


def fit(checkpoint_dir=None):
    model = UNet(
        input_channels          = TrainingConfig.INPUT_SHAPE[0],
        output_channels         = TrainingConfig.INPUT_SHAPE[0],
        base_channels           = ModelConfig.BASE_CH,
        base_channels_multiples = ModelConfig.BASE_CH_MULT,
        apply_attention         = ModelConfig.APPLY_ATTENTION,
        dropout_rate            = ModelConfig.DROPOUT_RATE,
        time_multiple           = ModelConfig.TIME_EMB_MULT,
    )
    if checkpoint_dir is not None:
        print('load weight:', checkpoint_dir)
        model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "ckpt.tar"), map_location='cpu')['model'])
    model.to(BaseConfig.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=TrainingConfig.LR)
    dataset = get_dataset(dataset_names=BaseConfig.DATASET)
    dataloader = get_dataloader(
        dataset  = dataset,
        batch_size    = TrainingConfig.BATCH_SIZE,
        device        = BaseConfig.DEVICE,
        pin_memory    = True,
        num_workers   = TrainingConfig.NUM_WORKERS,
    )

    loss_fn = nn.MSELoss()

    sd = SimpleDiffusion(
        num_diffusion_timesteps = TrainingConfig.TIMESTEPS,
        img_shape               = TrainingConfig.INPUT_SHAPE,
        device                  = BaseConfig.DEVICE,
    )

    scaler = amp.GradScaler()
    total_epochs = TrainingConfig.NUM_EPOCHS + 1
    log_dir, checkpoint_dir = setup_log_directory(config=BaseConfig())
    for epoch in range(1, total_epochs):
        torch.cuda.empty_cache()
        gc.collect()

        # Algorithm 1: Training
        train_one_epoch(model, sd, dataloader, optimizer, scaler, loss_fn, epoch=epoch)

        if epoch % 20 == 0:
            save_path = os.path.join(log_dir, f"{epoch}.png")

            # Algorithm 2: Sampling
            reverse_diffusion(model, sd, timesteps=TrainingConfig.TIMESTEPS, num_images=4,
                save_path=save_path, img_shape=TrainingConfig.INPUT_SHAPE, device=BaseConfig.DEVICE,
            )

            # clear_output()
            checkpoint_dict = {
                "opt": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint_dict, os.path.join(checkpoint_dir, "ckpt.tar"))
            print('{epoch} th Epoch:: Checkpoint saved :', os.path.join(checkpoint_dir, "ckpt.tar"))
            del checkpoint_dict


def train_one_epoch(model, sd, loader, optimizer, scaler, loss_fn, epoch=800,
                   base_config=BaseConfig(), training_config=TrainingConfig()):

    loss_record = MeanMetric()
    model.train()

    with tqdm(total=len(loader), dynamic_ncols=True) as tq:
        tq.set_description(f"Train :: Epoch: {epoch}/{training_config.NUM_EPOCHS}")

        for x0s in loader:
            tq.update(1)
            ts = torch.randint(low=1, high=training_config.TIMESTEPS, size=(x0s.shape[0],), device=base_config.DEVICE)
            xts, gt_noise = forward_diffusion(sd, x0s, ts)
            with amp.autocast():
                pred_noise = model(xts, ts)
                loss = loss_fn(gt_noise, pred_noise)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()

            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

            loss_value = loss.detach().item()
            loss_record.update(loss_value)

            tq.set_postfix_str(s=f"Loss: {loss_value:.4f}")

        mean_loss = loss_record.compute().item()

        tq.set_postfix_str(s=f"Epoch Loss: {mean_loss:.4f}")

    return mean_loss

@torch.inference_mode()
def reverse_diffusion(model, sd, timesteps=1000, img_shape=(3, 64, 64),
                      num_images=5, nrow=8, device="cpu", **kwargs):

    x = torch.randn((num_images, *img_shape), device=device)
    model.eval()

    for time_step in tqdm(iterable=reversed(range(1, timesteps)),
                          total=timesteps-1, dynamic_ncols=False,
                          desc="Sampling :: ", position=0):

        ts = torch.ones(num_images, dtype=torch.long, device=device) * time_step
        z = torch.randn_like(x) if time_step > 1 else torch.zeros_like(x)

        predicted_noise = model(x, ts)

        beta_t                            = get(sd.beta, ts)
        one_by_sqrt_alpha_t               = get(sd.one_by_sqrt_alpha, ts)
        sqrt_one_minus_alpha_cumulative_t = get(sd.sqrt_one_minus_alpha_cumulative, ts)

        x = (
            one_by_sqrt_alpha_t
            * (x - (beta_t / sqrt_one_minus_alpha_cumulative_t) * predicted_noise)
            + torch.sqrt(beta_t) * z
        )

    x = x.squeeze().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.plot(np.arange(x.shape[1]), x.T)
    plt.savefig(kwargs['save_path'], dpi=300, bbox_inches="tight")
    print('reverse diffusion plot saved to', kwargs['save_path'])
    return None

if __name__ == '__main__':
    demo_save_dir = os.path.join('/home/yainoue/meg2image/results', 'dp', 'demo')
    os.makedirs(demo_save_dir, exist_ok=True)
    # os.makedirs(BaseConfig.root_log_dir, exist_ok=True)
    # os.makedirs(BaseConfig.root_checkpoint_dir, exist_ok=True)

    difussion_demo(demo_save_dir, device=BaseConfig.DEVICE)
    checkpoint_dir = '/home/yainoue/meg2image/results/dp/checkpoints/version_6'
    fit(checkpoint_dir)