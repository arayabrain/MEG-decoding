import pandas as pd
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import DataLoader
import torch

import sys
sys.path.append('..')
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models import get_model_and_trainer


class EvalSettings:
    trained_sbj = ['1', '3', '1_3']
    eval_sbj = ['1', '2', '3']
    device = 'cuda'
    fs = 1000
    duration = 200
    roi = 'vc'
    # fss = [1000]
    # durations = [200]
    model_names = ['best', 'last'] # 'last'
    result_root = '../../results_ssl/'
    ckpt_pattern = 'sbj{sbj_name}/scmbm_{patch_size}-fs{fs}-dura{duration}/ckpt/{model_name}.pth'

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


    def __init__(self, sbj_name:str, roi, fs, duration):
        self.dataset_name =self.eval_dataset_name_dict[sbj_name]
        self.total_limit = 100
        self.h5_root = self.h5_root_dict[sbj_name].format(roi=roi, fs=fs, duration=duration)
        self.dataset_yaml = self.dataset_yaml_names_dict[f'{sbj_name}-{roi}']

        with initialize(config_path="../meg_ssl/ssl_configs/preprocess"):
            self.preprocess = compose(config_name='base')



def get_config(config_name):
    with initialize(config_path="../meg_ssl/ssl_configs"):
        cfg = compose(config_name=config_name)
    print('read config: ', config_name)

    cfg.training.logdir = None
    cfg.training.ckpt_dir = None
    cfg.training.reconst_fig_dir = None
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

    return dataset_dict['val']


def get_model_and_trainer(cfg):
    meg_encoder, trainer = get_model_and_trainer(cfg, device_count=1, usewandb=False, only_model=False)
    return meg_encoder, trainer



def first_setting(cfg):
    # get dataset
    train_dataset, val_dataset = get_dataset(cfg)

    # cfg tuning
    cfg.model.parameters.time_len = int(np.floor(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate))
    cfg.model.parameters.in_chans = train_dataset.datasets[0].num_electrodes

    # get model
    meg_encoder, trainer = get_model_and_trainer(cfg)
    return cfg



def eval_one_condition(dataloader, model, trainer, device):
    model.eval()
    model.to(device)
    trainer.device = device
    trainer.model = model
    trainer.val_loader = dataloader

    # evaluation
    print('=================start evaluation================')
    ret = trainer.validation()
    print('VAL/ loss:{:.3f},  acc:{:.3f}'.format(ret['val_loss'], ret['val_acc']))
    return ret




def run(settings):
    roi = settings.roi # 'vc'
    fs = settings.fs #1000
    dura = settings.duration # 200
    device = settings.device

    # get model
    model_base_cfg = get_config('sbj1_1k') # dummy
    model_base_cfg = first_setting(model_base_cfg)
    model, trainer = get_model_and_trainer(model_base_cfg)

    df= {'eval_sbj': [], 'trained_sbj': [], 'model_name': [], 'val_loss': [], 'val_acc': []}
    for e_sub in settings.eval_sbj:
        # get dataset for validation
        eval_sub_cfg = CFGBase(e_sub, roi, fs, dura)
        val_dataset = get_dataset(eval_sub_cfg)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
        for t_sub in settings.trained_sbj:
            for model_name in settings.model_names:
                #  'sbj{sbj_name}/scmbm_{patch_size}-fs{fs}-dura{duration}/ckpt/{model_name}.pth'
                ckpt_filename = settings.ckpt_pattern.format(sbj_name=t_sub, patch_size=4, fs=fs, duration=dura, model_name=model_name)
                ckpt_path = os.path.join(settings.result_root, ckpt_filename)
                print('load weight from ', ckpt_path)
                model = model.load_state_dict(torch.load(ckpt_path))
                ret = eval_one_condition(val_dataloader, model, trainer, device)

                df['eval_sbj'].append(e_sub)
                df['trained_sbj'].append(t_sub)
                df['model_name'].append(model_name)
                for key, value in ret.items():
                    df[key] = value

    result_save_dir = os.path.join(settings.result_root, 'vis_scalings')
    os.makedirs(result_save_dir, exist_ok=True)
    savepath = os.path.join(result_save_dir, '{roi}-{fs}-{dura}-{sbj_list}.csv'.format(roi=roi, fs=fs, dura=dura, sbj_list='_'.join(settings.eval_sbj)))

    df = pd.DataFrame(df)
    df.to_csv(savepath)
    print('DataFrame is saved to ', savepath)


if __name__ == '__main__':
    settings = EvalSettings()
    run(settings)