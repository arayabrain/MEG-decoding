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
from meg_ssl.utils.commons import set_seed
from hydra import compose, initialize
from meg_ssl.dataclass import parse_dataset
from transformers import AutoProcessor
from meg_ssl.utils.image_preprocess import numpy2image
from meg_ssl.trainers.diffusion_trainer import DiffusionTrainer
# vit_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
from torch.utils.data import ConcatDataset


class DiffusionDataset():
    def __init__(self, dataset, img_transform, ret_image_label=True):
        self.dataset = dataset
        self.img_transform = img_transform
        self.vit_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

        # DEBUG::
        self.ret_image_label = ret_image_label
        
        if ret_image_label:
            if isinstance(self.dataset, ConcatDataset):
                for ds in self.dataset.datasets:
                    if hasattr(ds, 'ret_image_label'):
                        ds.ret_image_label=ret_image_label


    def __len__(self):
        return len(self.dataset)

    @property
    def datasets(self):
        return self.dataset.datasets

    def __getitem__(self, idx):
        ret = {}
        if self.ret_image_label:
            tuple_data = self.dataset[idx]
            assert isinstance(tuple_data, tuple)
            if len(tuple_data)==3:
                eeg, image, label = tuple_data
            else:
                eeg, image = tuple_data
                label = -1 # dummy
        else:
            eeg, image = self.dataset[idx]
            label = -1 # dummy
        image_raw = numpy2image(image)
        image_raw = self.vit_processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        ret['image_raw'] = image_raw
        ret['image'] = self.img_transform(image.astype(np.float32)/255.0)
        ret['eeg'] = torch.from_numpy(eeg)
        ret['label'] = label
        

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
    parser.add_argument('--wandb_key_path', type=str, default=None)
    parser.add_argument('--device_counts', type=int, default=1)
    parser.add_argument('--ldf_exp', type=str, default='test')
    parser.add_argument('--datadir', type=str, default='sbj1')

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser.parse_args()

def update_config(args, config, exp_name):
    for attr in config.__dict__:
        if hasattr(args, attr):
            if getattr(args, attr) != None:
                setattr(config, attr, getattr(args, attr))

    config.logdir = config.logdir.format(exp_name=exp_name)
    config.ckpt_dir = config.ckpt_dir.format(exp_name=exp_name)
    config.reconst_dir = config.reconst_dir.format(exp_name=exp_name)
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
    # import pdb; pdb.set_trace()
    crop_pix = int(cfg.training.crop_ratio*cfg.training.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(cfg.training.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        channel_last
    ])
    return DiffusionDataset(dataset_dict['train'], img_transform_train), DiffusionDataset(dataset_dict['val'], img_transform_test)

def image_reconstruction(model, dataset:torch.utils.data.Dataset, state, num_samples:int, 
                         savedir:str, device:str='cuda'):
    model.to(device).eval()
    # import pdb; pdb.set_trace()
    for i, batch in enumerate(dataset):
        # import pdb; pdb.set_trace()
        dict_batch = {
            'eeg': torch.stack([batch['eeg'].to(device)]),
            'image': torch.stack([batch['image'].to(device)]),
            'label':torch.from_numpy(np.array([batch['label']]).astype(np.float32))
        }

        # grid, array, state = model.generate(dict_batch, 
        #                                     num_samples=num_samples, 
        #                                     ddim_steps=50, 
        #                                     HW=None, 
        #                                     limit=5, 
        #                                     state=state)
    
        grid, array, state = model.generate(dict_batch,
                                            ddim_steps=250,
                                            num_samples=num_samples,
                                            limit=5)
        
        savefile_path = os.path.join(savedir, f'{i}.png')
        # print(grid.shape, grid.max(), grid.min())
        grid = grid.astype(np.uint8)
        image = Image.fromarray(grid)
        image.save(savefile_path)
        print('grid is saved as ', savefile_path)
        array = array.squeeze().squeeze()
        print(array.shape, array.max(), array.min())
        if array.ndim == 4:
            array = np.transpose(array, (0, 2,3,1))
        else:
            array = np.transpose(array[0], (0, 2,3,1))
        for j, img in enumerate(array):
            image = Image.fromarray(img)
            img_name = str(j-1) if j > 0 else 'gt'
            image.save(os.path.join(savedir, f'{i}-{img_name}.png'))
        # if i > 10:
        #   break
        


def main(config, args):
    train_dataset, val_dataset = get_dataset(config)
    # import pdb; pdb.set_trace()
    num_voxels = int(meg_cfg.preprocess.meg_duration * meg_cfg.preprocess.brain_resample_rate) # eeg_latents_dataset_train.datasets[0].num_electrodes
    meg_encoder_pretrained_path = os.path.join(meg_cfg.meg_encoder_path.format(sbj_name=args.datadir, exp_name=args.meg_exp))
    meg_cfg.meg_encoder.parameters.in_chans = val_dataset.datasets[0].num_electrodes
    pretrain_mbm_metafile = {
        'model':torch.load(meg_encoder_pretrained_path),
        'config': meg_cfg.meg_encoder.parameters
    }
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # create generateive model
    eldm = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=None,
                ddim_steps=config.training.ddim_steps, global_pool=config.training.global_pool, 
                use_time_cond=config.training.use_time_cond,
                clip_tune = config.training.clip_tune, cls_tune = config.training.cls_tune)
    generative_model = eldm.model

    # model settings
    generative_model.main_config = config
    generative_model.output_path = config.output_path
    generative_model.run_full_validation_threshold = 0.15
    generative_model.learning_rate = config.training.lr
    generative_model.eval_avg = config.training.eval_avg

    state = None
    # # load model weight
    # weightpath = os.path.join(f'../../results_task/dream_diffusion/{args.meg_exp}-{args.ldf_exp}/ckpt/current-generator.pth')
    # model_info = torch.load(weightpath, map_location='cpu')
    # print('pretrained model is loaded from ', weightpath)
    # before_weight = copy.deepcopy(generative_model.cpu().state_dict())
    # print('generative_model last layer summation: ', generative_model.state_dict()['image_embedder.transformer.visual_projection.weight'].sum(), get_total_mean(generative_model.state_dict()))
    # state = None
    # if 'model_state_dict' in model_info.keys():
    #     generative_model.load_state_dict(model_info['model_state_dict'])
    #     state = model_info['state']
    # else:
    #     generative_model.load_state_dict(model_info) # (model_info)
    #     state = None
    # ## DEBUG:::::::::::::
    generative_model.load_state_dict(torch.load('../../results_task/dream_diffusion/scmbm_4-fs1000-dura200-test_enc-6k/ckpt/current-generator.pth'))
    generative_model.cond_stage_model.load_state_dict(torch.load('../../results_task/dream_diffusion/scmbm_4-fs1000-dura200-test_enc-6k/ckpt/current-meg_enc.pth'))

    print('generative_model last layer summation: ', generative_model.state_dict()['image_embedder.transformer.visual_projection.weight'].sum(), get_total_mean(generative_model.state_dict()))
    # compare_weight(before_weight, generative_model.cpu().state_dict())
    # print('=============== before vs model_info==========')
    # compare_weight(before_weight, model_info)
    # import pdb; pdb.set_trace()
    # exit()

    # trainable settings
    generative_model.freeze_whole_model()
    generative_model.freeze_first_stage()
    # generative_model.freeze_whole_model()
    # generative_model.unfreeze_cond_stage()
    generative_model.train_cond_stage_only = False # True
    
    # generate
    savedir = os.path.join(f'../../results_task/dream_diffusion/{args.meg_exp}-{args.ldf_exp}/reconst/eval')
    os.makedirs(savedir, exist_ok=True)
    num_samples = 3
    # import pdb; pdb.set_trace()
    # assert len(val_dataset) == 1200
    val_dataset[0] # image_namesのindex errorがきが
    image_reconstruction(generative_model, val_dataset, state, num_samples, savedir)
    print('end')

def get_total_mean(state):
    total_mean = 0
    for k, s in state.items():
        total_mean += s.sum().item()
    return total_mean

def compare_weight(state1, state2):
    for k in list(state1.keys()):
        s1 = state1[k]
        s2 = state2[k]
        if  not (s1==s2).cpu().numpy().all():
            continue
        print(k, (s1==s2).cpu().numpy().all())# , s1.sum(), s2.sum())
     


if __name__ == '__main__':
    args = get_args_parser()

    set_seed(49)
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

    exp_name = args.meg_exp + '-' + args.ldf_exp
    print('experiment name is ', exp_name)
    meg_cfg.training = update_config(args, meg_cfg.training, exp_name)
    meg_cfg.output_path = meg_cfg.output_path.format(exp_name=exp_name)
    main(meg_cfg, args)