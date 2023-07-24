import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from termcolor import cprint
# import wandb
import pandas as pd
from omegaconf import DictConfig, open_dict
import hydra
from hydra.utils import get_original_cwd

from constants import device
# from speech_decoding.dataclass.brennan2018 import Brennan2018Dataset
# from speech_decoding.dataclass.gwilliams2022 import (
#     Gwilliams2022SentenceSplit,
#     Gwilliams2022ShallowSplit,
#     Gwilliams2022DeepSplit,
#     Gwilliams2022Collator,
# )
from torch.utils.data import DataLoader, RandomSampler, BatchSampler

from meg_decoding.models import get_model, Classifier
from meg_decoding.utils.get_dataloaders import get_dataloaders, get_samplers
from meg_decoding.utils.loss import *
from meg_decoding.dataclass.god import GODDatasetBase, GODCollator
from meg_decoding.utils.loggers import Pickleogger
from meg_decoding.utils.vis_grad import get_grad
from torch.utils.data.dataset import Subset
import matplotlib.pyplot as plt
import seaborn as sns


def run(args: DictConfig, eval_sbj:str='1') -> None:

    from meg_decoding.utils.reproducibility import seed_worker
    # NOTE: We do need it (IMHO).
    if args.reproducible:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        g = torch.Generator()
        g.manual_seed(0)
        seed_worker = seed_worker
    else:
        g = None
        seed_worker = None

    pkl_logger = Pickleogger(os.path.join(args.save_root, 'runs'))

    # with open_dict(args):
    #     args.root_dir = get_original_cwd()
    cprint(f"Current working directory : {os.getcwd()}")
    cprint(args, color="white")

    # -----------------------
    #       Dataloader
    # -----------------------
    feature_layer = 'clip' if not 'feature_layer' in args.keys() else args['feature_layer']
    # source_dataset = GODDatasetBase(args, 'train', feature_layer=feature_layer)
    source_dataset = GODDatasetBase(args, 'train', return_label=True, feature_layer=feature_layer)
    outlier_dataset = GODDatasetBase(args, 'val', return_label=True,
                                        mean_X= source_dataset.mean_X,
                                        mean_Y=source_dataset.mean_Y,
                                        std_X=source_dataset.std_X,
                                        std_Y=source_dataset.std_Y,
                                        feature_layer=feature_layer
                                    )

    if eval_sbj == '1':
        ind_tr = list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
        ind_out = list(range(0,50))
    elif eval_sbj == '2':
        ind_tr = list(range(7200, 7200+3000)) + list(range(10800, 10800+3000))
        ind_te = list(range(7200+3000, 7200+3600)) + list(range(10800+3000, 10800+3600))
        ind_out = list(range(50,100))
    elif eval_sbj == '3':
        ind_tr = list(range(14400, 14400+3000)) + list(range(14400+3600, 14400+6600))
        ind_te = list(range(14400+3000,14400+3600)) + list(range(14400+6600, 14400+7200))
        ind_out = list(range(100,150))
    else:
        ind_tr = list(range(0, 3000)) + list(range(3600, 6600))  + list(range(7200, 7200+3000))  + list(range(10800, 10800+3000)) + list(range(14400, 14400+3000)) + list(range(14400+3600, 14400+6600))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200))  + list(range(7200+3000, 7200+3600)) + list(range(10800+3000, 10800+3600)) + list(range(14400+3000,14400+3600)) + list(range(14400+6600, 14400+7200))
        ind_out = list(range(0,150))

    outlier_dataset = Subset(outlier_dataset, ind_out)
    train_dataset = Subset(source_dataset, ind_tr)
    val_dataset   = Subset(source_dataset, ind_te)


    with open_dict(args):
        args.num_subjects = source_dataset.num_subjects
        print('num subject is {}'.format(args.num_subjects))


    if args.use_sampler:
        test_size = 50# 重複サンプルが存在するのでval_dataset.Y.shape[0]
        train_loader, test_loader = get_samplers(
            train_dataset,
            val_dataset,
            args,
            test_bsz=test_size,
            collate_fn=GODCollator(args),)

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size= args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        val_loader = DataLoader(
            val_dataset, #
            batch_size=50, # args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
        test_loader = DataLoader(
            outlier_dataset,  # val_dataset
            batch_size=50, # args.batch_size,
            drop_last=False,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(args.save_root, 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)


    classifier = Classifier(args)

    # ---------------
    #      Loss
    # ---------------
    loss_func = CLIPLoss(args).to(device)
    loss_func.eval()
    # ======================================
    train_losses = []
    test_losses = []
    trainTop1accs = []
    trainTop10accs = []
    testTop1accs = []
    testTop10accs = []
    train_Zs = []
    train_Ys = []
    train_Ls = []
    brain_encoder.eval()
    pbar2 = tqdm(train_loader)
    for i, batch in enumerate(pbar2):
        with torch.no_grad():
            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, Labels = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)
            # import pdb; pdb.set_trace()
            Z = brain_encoder(X, subject_idxs)
            train_Zs.append(Z)
            train_Ys.append(Y)
            train_Ls.append(Labels)

    train_Zs = torch.cat(train_Zs, dim=0).detach().cpu().numpy()
    train_Ys = torch.cat(train_Ys, dim=0).detach().cpu().numpy()
    train_Ls = torch.cat(train_Ls, dim=0).detach().cpu().numpy()
    z_savepath = os.path.join(args.save_root, 'features','z_train.npy')
    y_savepath = os.path.join(args.save_root, 'features', 'y_train.npy')
    l_savepath = os.path.join(args.save_root, 'features', 'l_train.npy')
    print('train:', train_Zs.shape)
    print('saved to: ', z_savepath)
    np.save(z_savepath, train_Zs)
    np.save(y_savepath, train_Ys)
    np.save(l_savepath, train_Ls)

    val_Zs = []
    val_Ys = []
    val_Ls = []
    brain_encoder.eval()
    for batch in val_loader:
        with torch.no_grad():

            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, Labels = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB
            val_Zs.append(Z)
            val_Ys.append(Y)
            val_Ls.append(Labels)

    val_Zs = torch.cat(val_Zs, dim=0).detach().cpu().numpy()
    val_Ys = torch.cat(val_Ys, dim=0).detach().cpu().numpy()
    val_Ls = torch.cat(val_Ls, dim=0).detach().cpu().numpy()
    z_savepath = os.path.join(args.save_root, 'features', 'z_val.npy')
    y_savepath = os.path.join(args.save_root, 'features', 'y_val.npy')
    l_savepath = os.path.join(args.save_root, 'features', 'l_val.npy')
    np.save(z_savepath, val_Zs)
    np.save(y_savepath, val_Ys)
    np.save(l_savepath, val_Ls)
    print('val: ', val_Zs.shape)


    test_Zs = []
    test_Ys = []
    test_Ls = []
    brain_encoder.eval()
    for batch in test_loader:
        with torch.no_grad():

            if len(batch) == 3:
                X, Y, subject_idxs = batch
            elif len(batch) == 4:
                X, Y, subject_idxs, Labels = batch
            else:
                raise ValueError("Unexpected number of items from dataloader.")

            X, Y = X.to(device), Y.to(device)

            Z = brain_encoder(X, subject_idxs)  # 0.96 GB
            test_Zs.append(Z)
            test_Ys.append(Y)
            test_Ls.append(Labels)


    test_Zs = torch.cat(test_Zs, dim=0).detach().cpu().numpy()
    test_Ys = torch.cat(test_Ys, dim=0).detach().cpu().numpy()
    test_Ls = torch.cat(test_Ls, dim=0).detach().cpu().numpy()
    z_savepath = os.path.join(args.save_root, 'features', 'z_test.npy')
    y_savepath = os.path.join(args.save_root, 'features', 'y_test.npy')
    l_savepath = os.path.join(args.save_root, 'features', 'l_test.npy')
    np.save(z_savepath, test_Zs)
    np.save(y_savepath, test_Ys)
    np.save(l_savepath, test_Ls)
    print('test: ', test_Zs.shape)
    plt.plot(np.arange(test_Zs.shape[1]), test_Zs[0])
    plt.plot(np.arange(test_Ys.shape[1]), test_Ys[0])
    plt.savefig(os.path.join(args.save_root, 'features', 'zy_test.png'))


if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        # args = compose(config_name='20230427_sbj01_eegnet')
        args = compose(config_name='20230429_sbj01_eegnet_regression')
        # args = compose(config_name='20230501_all_eegnet_regression')
        # args = compose(config_name='20230425_sbj01_seq2stat')
        args = compose(config_name='20230515_sbj02_eegnet_regression')
        # args = compose(config_name='20230516_sbj03_eegnet_regression')
        # args = compose(config_name='20230519_all_eegnet_regression_src_reconst')
        # args = compose(config_name='20230518_all_eegnet_regression')
        # args = compose(config_name='20230523_sbj01_eegnet_regression_src_reconst')
        # args = compose(config_name='20230524_all_eegnet_regression_src_reconst')
        # args = compose(config_name='20230531_sbj02_eegnet_regression_src_reconst')
        # args = compose(config_name='20230601_sbj03_eegnet_regression_src_reconst')
        # args = compose(config_name='20230606_sbj02_eegnet')
        # args = compose(config_name='20230607_sbj03_eegnet')
        args = compose(config_name = '20230621_sbj01_eegnet_regression_cnn')
        args = compose(config_name = '20230623_sbj01_eegnet_regression_cnn3')
        args = compose(config_name = '20230622_sbj01_eegnet_regression_cnn5')
        args = compose(config_name = '20230622_sbj01_eegnet_regression_cnn8')
        args = compose(config_name = '20230628_sbj03_eegnet_regression_cnn1')
        args = compose(config_name = '20230629_sbj03_eegnet_regression_cnn3')
        args = compose(config_name = '20230630_sbj03_eegnet_regression_cnn5')
        args = compose(config_name = '20230702_sbj03_eegnet_regression_cnn8')
    # for subset of 20230501
    # with initialize(version_base=None, config_path="../configs/subjects"):
    #     args.subjects = compose(config_name='pattern_sbj01')
    eval_sbj = '1'
    if not os.path.exists(os.path.join(args.save_root, 'fetures')):
        os.makedirs(os.path.join(args.save_root, 'features'), exist_ok=True)
    run(args, eval_sbj)
