import pandas as pd
import os
from hydra import compose, initialize
from omegaconf import OmegaConf
import numpy as np
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('.')
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models import get_model_and_trainer
plt.rcParams["font.size"] = 16

class EvalSettings:
    trained_sbj = ['1', '3', '1_3']
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
    n_samples_replace_dict = {
        '1': {'all': 31200, '-1k': 1000, '-2.5k': 2500, '-5k': 5000, '-10k':10000},
        '3': {'all': 31200, '-1k': 1000, '-2.5k': 2500, '-5k': 5000, '-10k':10000},
        '1_3': {'all': 62300, '-1k': 1000, '-2.5k': 2500, '-5k': 5000, '-10k':10000}
    }

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



def run(settings):
    roi = settings.roi # 'vc'
    fs = settings.fs #1000
    dura = settings.duration # 200
    device = settings.device

    result_save_dir = os.path.join(settings.result_root, 'vis_scalings')
    result_save_dir = os.path.join(settings.result_root, 'vis_scalings')
    csvpath = os.path.join(result_save_dir, '{roi}-{fs}-{dura}-{sbj_list}.csv'.format(roi=roi, fs=fs, dura=dura, sbj_list='_'.join(settings.eval_sbj)))
    df = pd.read_csv(csvpath)
    df = df.fillna('all')
    fig, axes = plt.subplots(nrows=2, ncols=len(settings.eval_sbj), figsize=(24, 12))
    for i, e_sub in enumerate(settings.eval_sbj):
        e_sub_df = df.query('eval_sbj=={}'.format(e_sub))
        for t_sub in settings.trained_sbj:
            t_sub_df = e_sub_df.query('trained_sbj=="{}"'.format(t_sub))
            # print(t_sub, settings.n_samples_replace_dict[t_sub])
            # import pdb; pdb.set_trace()
            t_sub_df = t_sub_df.replace({'n_sample': settings.n_samples_replace_dict[t_sub]})
            t_sub_df = t_sub_df.sort_values(by='n_sample', axis=0)
            # best
            # best_df = t_sub_df.query('model_name=="best"')
            # x = best_df['n_sample'].to_list()
            # loss = best_df['val_loss'].to_list()
            # corr = best_df['val_corr'].to_list()
            # axes[0, i].plot(x, loss, '-o', label=f'{t_sub}-best')
            # axes[1, i].plot(x, corr, '-o', label=f'{t_sub}-best')
            # import pdb; pdb.set_trace()
            # last
            last_df = t_sub_df.query('model_name=="last"')
            x = last_df['n_sample'].to_list()
            loss = last_df['val_loss'].to_list()
            corr = last_df['val_corr'].to_list()
            axes[0, i].plot(x, loss, label=f'{t_sub}-last')
            axes[1, i].plot(x, corr, label=f'{t_sub}-last')
        axes[0,i].legend()
        axes[1,i].legend()
        axes[0,i].set_ylabel('test loss (sbj: {})'.format(e_sub))
        axes[1,i].set_ylabel('test corr (sbj: {})'.format(e_sub))
        axes[0,i].set_xlabel('num samples (log)')
        axes[1,i].set_xlabel('num samples (log)')

        axes[0,i].set_xscale('log')
        axes[1,i].set_xscale('log')

    pngpath = os.path.join(csvpath.replace('.csv', 'last.png'))
    plt.savefig(pngpath,bbox_inches="tight")
    print('save image as ', pngpath)

if __name__ == '__main__':
    settings = EvalSettings()
    run(settings)