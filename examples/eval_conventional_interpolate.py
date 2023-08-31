import pandas as pd
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import DataLoader
import torch

import sys
sys.path.append('.')
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models import get_model_and_trainer
from meg_ssl.utils.conventional_interpolate import polynomial_interpolate, patchify, unpatchify, calc_loss_and_corr
import matplotlib.pyplot as plt

class EvalSettings:
    order = [1, 2, 3, 4]
    eval_sbj = ['1', '2', '3']
    num_samples = ['', '-10k', '-5k', '-2.5k', '-1k'] # roi等の情報も含める場合がある
    device = 'cuda'
    fs = 1000
    duration = 200
    roi = 'vc'
    # fss = [1000]
    # durations = [200]
    model_names = ['best', 'last'] # 'last'
    result_root = '../../results_ssl/'
    ckpt_pattern = 'sbj{sbj_name}/scmbm_{patch_size}-fs{fs}-dura{duration}{n_sample}/ckpt/{model_name}.pth'


class CFGBase():
    eval_dataset_name_dict ={
        '1': {'train':{'drama': 'sbj_1-session_1'},
              'val': {'GOD': 'sbj_1-train-session_6_12'}},
        '2': {'train':{'drama': 'sbj_1-session_1'}, # dummy
              'val': {'GOD': 'sbj_2-train-session_6_12'}},
        '3': {'train':{'drama': 'sbj_3-session_1'},
              'val': {'GOD': 'sbj_3-train-session_6_12'}}
    }
    h5_root_dict = {
        '1': '../../../dataset/ssl_dataset/sbj1/{roi}-fs{fs}-dura{duration}-1',
        '2': '../../../dataset/ssl_dataset/sbj2/', # None
        '3': '../../../dataset/ssl_dataset/sbj3/sbj3-{roi}-fs{fs}-dura{duration}-1'
    }
    dataset_yaml_names_dict = {
        '1-vc': {'drama': 'drama/drama_vc.yaml', 'GOD': 'GOD/god_vc.yaml'},
        '2-vc': {'drama': 'drama/drama_vc.yaml', 'GOD': 'GOD/god_vc.yaml'},
        '3-vc': {'drama': 'drama/drama_vc.yaml', 'GOD': 'GOD/god_vc.yaml'},
    }
    total_limit_dict = {
        '1': {'train': {'drama': 100}, 'val': {'GOD': 1200}},
        '2': {'train': {'drama': 100}, 'val': {'GOD': 1200}},
        '3': {'train': {'drama': 100}, 'val': {'GOD': 1200}},
    }


    def __init__(self, sbj_name:str, roi, fs, duration):
        self.dataset_name =self.eval_dataset_name_dict[sbj_name]
        self.total_limit = self.total_limit_dict[sbj_name]

        self.h5_root = self.h5_root_dict[sbj_name].format(roi=roi, fs=fs, duration=duration)
        self.dataset_yaml = self.dataset_yaml_names_dict[f'{sbj_name}-{roi}']

        with initialize(config_path="../meg_ssl/ssl_configs/preprocess"):
            self.preprocess = compose(config_name=f'fs{fs}_dura{duration}')


def get_config(config_name):
    with initialize(config_path="../meg_ssl/ssl_configs"):
        cfg = compose(config_name=config_name)
    print('read config: ', config_name)

    cfg.training.logdir = 'tmps/eval/log_dummy'
    cfg.training.ckpt_dir = 'tmps/eval/ckpt_dummy'
    cfg.training.reconst_fig_dir = 'tmps/eval/reconst_dummy'
    return cfg


def get_dataset(cfg):
    dataset_names:dict = cfg.dataset_name
    # import pdb; pdb.set_trace()
    dataset_yamls:dict = cfg.dataset_yaml
    num_trial_limit:dict = cfg.total_limit
    preproc_config:OmegaConf = cfg.preprocess
    h5_root:str = cfg.h5_root
    image_preprocs:list = []
    meg_preprocs:list = []
    only_meg:bool = True
    on_memory:bool = True #False
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit,
                                      h5_root, image_preprocs, meg_preprocs, only_meg, on_memory)

    return dataset_dict['train'], dataset_dict['val']


def get_model_and_trainer_from_cfg(cfg):
    meg_encoder, trainer = get_model_and_trainer(cfg, device_count=1, usewandb=False, only_model=False)
    return meg_encoder, trainer



def first_setting(cfg, fs, duration, manual_setting=True):
    with initialize(config_path="../meg_ssl/ssl_configs/preprocess"):
        cfg.preprocess = compose(config_name=f'fs{fs}_dura{duration}')

    if manual_setting:
        cfg.model.parameters.time_len = 208
        cfg.model.parameters.in_chans = 22
    else:
        # get dataset
        train_dataset, _ = get_dataset(cfg)

        # cfg tuning
        cfg.model.parameters.time_len = int(np.floor(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate))
        cfg.model.parameters.in_chans = train_dataset.datasets[0].num_electrodes
        print('first settings: time_len', cfg.model.parameters.time_len)
        print('first settings: in_chans', cfg.model.parameters.in_chans)
    return cfg



def eval_one_condition(dataloader, model, config, order, save_image=False):
    mask_ratio = config.training.mask_ratio
    patch_size = config.model.parameters.patch_size
    loss_list = []
    corr_list = []
    for i, batch in enumerate(dataloader):
        data = batch
        _, mask, _ = model.random_masking(torch.zeros([data.shape[0], int(data.shape[2]/4), 1024]), mask_ratio)

        data = data.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        pred_data = polynomial_interpolate(data, mask, patch_size, order)
        loss, corr = calc_loss_and_corr(data, pred_data, mask, patch_size)
        loss_list.append(loss)
        corr_list.append(corr)
        if save_image:
            if i > 0:
                continue
            unpatch_data = data # unpatchify(data, patch_size)
            unpatch_pred = pred_data # unpatchify(pred_data, patch_size)
            fig, axes = plt.subplots(nrows=5, figsize=(10, 10))
            for c in range(5):
                axes[c].plot(np.arange(len(unpatch_data[0,0,:])), unpatch_data[0,c], label='original')
                axes[c].plot(np.arange(len(unpatch_data[0,0,:])), unpatch_pred[0,c], label='interpolated')
                axes[c].legend()
            plt.savefig('tmp_conventional_reconst.png')

    ret = {'val_loss': np.mean(loss_list), 'val_corr': np.mean(corr_list)}

    # evaluation
    print('=================start evaluation================')
    print('VAL/ loss:{:.3f},  corr:{:.3f}'.format(ret['val_loss'], ret['val_corr']))
    return ret




def run(settings):
    roi = settings.roi # 'vc'
    fs = settings.fs #1000
    dura = settings.duration # 200
    device = settings.device

    # get model
    model_base_cfg = get_config('sbj1_1k') # dummy
    model_base_cfg = first_setting(model_base_cfg, fs, dura)
    model, trainer = get_model_and_trainer_from_cfg(model_base_cfg)

    df= {'eval_sbj': [], 'order': [], 'model_name': [], 'val_loss': [], 'val_corr': []}
    for e_sub in settings.eval_sbj:
        # get dataset for validation
        eval_sub_cfg = CFGBase(e_sub, roi, fs, dura)
        train_dataset, val_dataset = get_dataset(eval_sub_cfg)
        # import pdb; pdb.set_trace()
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=1)
        for order in settings.order:
            for model_name in settings.model_names:
                #  'sbj{sbj_name}/scmbm_{patch_size}-fs{fs}-dura{duration}/ckpt/{model_name}{n_sample}.pth'
                ret = eval_one_condition(val_dataloader, model, model_base_cfg, order, save_image=True)

                df['eval_sbj'].append(e_sub)
                df['order'].append(order)
                df['model_name'].append(model_name)
                for key, value in ret.items():
                    df[key].append(value)


    result_save_dir = os.path.join(settings.result_root, 'vis_scalings')
    os.makedirs(result_save_dir, exist_ok=True)
    savepath = os.path.join(result_save_dir, 'conventional-{roi}-{fs}-{dura}-{sbj_list}.csv'.format(roi=roi, fs=fs, dura=dura, sbj_list='_'.join(settings.eval_sbj)))

    df = pd.DataFrame(df)
    df.to_csv(savepath)
    print('DataFrame is saved to ', savepath)


if __name__ == '__main__':
    settings = EvalSettings()
    run(settings)