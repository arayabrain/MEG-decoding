import os, sys
import numpy as np
import torch
import argparse
import datetime
import wandb
import torchvision.transforms as transforms
from einops import rearrange
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import copy

# own code
from meg_ssl.generate_configs.config import Config_Generative_Model
# from dataset import  create_EEG_dataset
from meg_ssl.models.dc_ldm.ldm_for_eeg import eLDM
from meg_ssl.utils.diffusion_utils.eval_metrics import get_similarity_metric
from meg_ssl.utils.commons import set_seed, get_device_count, get_device
from hydra import compose, initialize
from meg_ssl.dataclass import parse_dataset
from transformers import AutoProcessor
from meg_ssl.utils.image_preprocess import numpy2image
from meg_ssl.trainers import DiffusionTrainer
# vit_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")



class DiffusionDataset():
    def __init__(self, dataset, img_transform):
        self.dataset = dataset
        self.img_transform = img_transform
        self.vit_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

    def __len__(self):
        return len(self.dataset)

    @property
    def datasets(self):
        return self.dataset.datasets

    def __getitem__(self, idx):
        ret = {}
        eeg, image = self.dataset[idx]
        image_raw = numpy2image(image)
        image_raw = self.vit_processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        ret['image_raw'] = image_raw
        ret['image'] = self.img_transform(image.astype(np.float32)/255.0)
        ret['eeg'] = torch.from_numpy(eeg)
        ret['label'] = 0 # dummy

        return ret


def get_args_parser():
    parser = argparse.ArgumentParser('Double Conditioning LDM Finetuning', add_help=False)
    # project parameters
    parser.add_argument('--seed', type=int)
    parser.add_argument('--root_path', type=str, default = '../dreamdiffusion/')
    parser.add_argument('--pretrain_mbm_path', type=str)
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--crop_ratio', type=float)
    parser.add_argument('--dataset', type=str)

    # finetune parameters
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--precision', type=int)
    parser.add_argument('--accumulate_grad', type=int)
    parser.add_argument('--global_pool', type=bool)

    # diffusion sampling parameters
    parser.add_argument('--pretrain_gm_path', type=str)
    parser.add_argument('--num_samples', type=int)
    parser.add_argument('--ddim_steps', type=int)
    parser.add_argument('--use_time_cond', type=bool)
    parser.add_argument('--eval_avg', type=bool)

    # meg specific parameters
    parser.add_argument('--meg_config', type=str, default='ssl/ssl_configs/test_config.yaml')
    parser.add_argument('--meg_encoder', type=str, default=None)
    parser.add_argument('--meg_preprocess', type=str, default=None)
    parser.add_argument('--meg_exp', type=str, default=None)
    parser.add_argument('--meg_h5name', type=str, default=None)
    parser.add_argument('--wandb_key_path', type='str', default=None)
    parser.add_argument('--device_counts', type=int, default=1)

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser.parse_args()

def update_config(args, config):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))
    return config

def create_readme(config, path):
    print(config.__dict__)
    with open(os.path.join(path, 'README.md'), 'w+') as f:
        print(config.__dict__, file=f)


def normalize(img):
    if img.shape[-1] == 3:
        img = rearrange(img, 'h w c -> c h w')
    img = torch.tensor(img)
    img = img * 2.0 - 1.0 # to -1 ~ 1
    return img

class random_crop:
    def __init__(self, size, p):
        self.size = size
        self.p = p
    def __call__(self, img):
        if torch.rand(1) < self.p:
            return transforms.RandomCrop(size=(self.size, self.size))(img)
        return img

def fmri_transform(x, sparse_rate=0.2):
    # x: 1, num_voxels
    x_aug = copy.deepcopy(x)
    idx = np.random.choice(x.shape[0], int(x.shape[0]*sparse_rate), replace=False)
    x_aug[idx] = 0
    return torch.FloatTensor(x_aug)


def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_dataset(cfg):
    dataset_names:dict = cfg.dataset_name
    # import pdb; pdb.set_trace()
    dataset_yamls:dict = cfg.dataset_yaml
    num_trial_limit:dict = cfg.total_limit
    preproc_config = cfg.preprocess
    h5_root:str = cfg.h5_root
    image_preprocs:list = []
    meg_preprocs:list = []
    only_meg:bool = False
    on_memory:bool = False
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit,
                                    h5_root, image_preprocs, meg_preprocs, only_meg, on_memory)
    crop_pix = int(cfg.crop_ratio*cfg.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(cfg.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        channel_last
    ])
    return DiffusionDataset(dataset_dict['train'], img_transform_train), DiffusionDataset(dataset_dict['val'], img_transform_test)


def main(config, args):
    train_dataset, val_dataset = get_dataset(config)
    num_voxels = int(meg_cfg.preprocess.meg_duration * meg_cfg.preprocess.brain_resample_rate) # eeg_latents_dataset_train.datasets[0].num_electrodes
    meg_encoder_pretrained_path = os.path.join(meg_cfg.meg_encoder_path.format(exp_name=args.meg_exp))
    meg_cfg.meg_encoder.parameters.in_chans = val_dataset.datasets[0].num_electrodes
    pretrain_mbm_metafile = {
        'model':torch.load(meg_encoder_pretrained_path),
        'config': meg_cfg.meg_encoder.parameters
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond,
                clip_tune = config.clip_tune, cls_tune = config.cls_tune)
    # diffusion trainer
    if args.wandb_key_path is not None:
        usewandb=True
        with open(args.wandb_key_path, 'r') as f:
            wandb_key = f.read()
            wandb_key = wandb_key.split('\n')[0]
        wandb.login(key = f'{wandb_key}')
        # wandb.init(project='llm_hackason-'+args.config)
        # wandb.init(project='llm_hackason-new_prompt')
        wandb.init(project='meg-god')
    else:
        usewandb=False
    trainer = DiffusionTrainer(config, args.device_counts, usewandb)
    # fit
    trainer.fit(generative_model, [], train_dataset, val_dataset)



if __name__ == '__main__':
    args = get_args_parser()

    set_seed(42)
    with initialize(config_path='meg_ssl/generate_configs/'):
        meg_cfg = compose(args.meg_config)
    if args.meg_encoder is not None:
        with initialize(config_path='meg_ssl/ssl_configs/model'):
            meg_cfg.meg_encoder = compose(args.meg_encoder)
        print('INFO ========= model config is overrided by ', args.meg_encoder)
    if args.meg_preprocess is not None:
        with initialize(config_path='meg_ssl/ssl_configs/preprocess'):
            meg_cfg.preprocess = compose(args.meg_preprocess)
        print('INFO ========= preprocess config is overrided by', args.meg_preprocess)
    # num_electrodes, fs, bpがh5ファイルに関係している
    if args.meg_h5name is None:
        meg_cfg.h5_root = meg_cfg.h5_root.format(h5_name='fs{}-bp{}_{}'.format(meg_cfg.preprocess.brain_resample_rate, *meg_cfg.preprocess.bandpass_filter))
    else:
        meg_cfg.h5_root = meg_cfg.h5_root.format(h5_name=args.meg_h5name)
    meg_cfg.training = update_config(args, meg_cfg.training)
    main(meg_cfg, args)