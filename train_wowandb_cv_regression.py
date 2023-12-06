import os, sys, random
import numpy as np
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm, trange
from termcolor import cprint
# import wandb

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
from meg_ssl.dataclass import parse_dataset
from omegaconf import OmegaConf
from meg_ssl.models.image_encoder import get_image_encoder
import wandb
from meg_ssl.utils.image_preprocess import numpy2image


def get_dataset(cfg:OmegaConf):
    dataset_names:dict = cfg.dataset_name
    dataset_yamls:dict = cfg.dataset_yaml
    num_trial_limit:dict = cfg.total_limit
    preproc_config:OmegaConf = cfg.preprocess
    h5_root:str = cfg.h5_root
    image_preprocs:list = []
    meg_preprocs:list = []
    only_meg:bool = False
    on_memory:bool = True #False
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit, 
                                      h5_root, image_preprocs, meg_preprocs, only_meg, on_memory)
    
    return dataset_dict['train'], dataset_dict['val']
    


def run(args: DictConfig) -> None:
    wandb_key_path = args.wandb_key_path
    if wandb_key_path is not None:
        usewandb=True
        with open(wandb_key_path, 'r') as f:
            wandb_key = f.read()
            wandb_key = wandb_key.split('\n')[0]
        wandb.login(key = f'{wandb_key}')
        # wandb.init(project='llm_hackason-'+args.config)
        # wandb.init(project='llm_hackason-new_prompt')
        wandb.init(project='meg-image-clip')        
    else:
        usewandb=False

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

    with initialize(config_path='meg_ssl/task_configs/model'):
        args.image_encoder = compose(args.image_encoder)
    with initialize(config_path='meg_ssl/ssl_configs/preprocess'):
        args.preprocess = compose(args.preprocess)

    image_encoder, image_processor = get_image_encoder(args.image_encoder.name, args.image_encoder.parameters)
    image_encoder.eval()
    # ---------------------
    #        Dataloader
    # ---------------------
    train_dataset, val_dataset = get_dataset(args)
    def collate_fn(batch):
        # create new batch
        # B x 1 x electrodes x time
        batch_meg = torch.stack([torch.from_numpy(b[0]) for b in batch])
        # batch_image = [numpy2image(b[1]) for b in batch]
        # batch_image = transform2vit_image_only_inputs(batch_image)
        # batch_image = image_processor(batch_image['text'], batch_image['images'], return_tensors="pt", padding=True)
        batch_image = torch.stack([image_processor(numpy2image(b[1])) for b in batch])
        return batch_meg, batch_image
    
    train_loader = DataLoader(
                train_dataset,
                batch_size= args.batch_size,
                drop_last=True,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
                collate_fn=collate_fn
            )
    test_loader = DataLoader(
        val_dataset,
        batch_size=50, # args.batch_size,
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
        collate_fn=collate_fn
    )

    # if args.use_wandb:
    #     wandb.config = {k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]}
    #     wandb.init(
    #         project=args.wandb.project,
    #         entity=args.wandb.entity,
    #         config=wandb.config,
    #         save_code=True,
    #     )
    #     wandb.run.name = args.wandb.run_name + "_" + args.split_mode
    #     wandb.run.save()

    # normalize_mean_X = torch.from_numpy(np.load('/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/stats/mean_X.npy')).to(device)
    normalize_mean_Y = torch.from_numpy(np.load('/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/stats/mean_Y.npy')).to(device)
    # normalize_std_X = torch.from_numpy(np.load('/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/stats/std_X.npy')).to(device)
    normalize_std_Y = torch.from_numpy(np.load('/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/stats/std_Y.npy')).to(device)

    # ---------------------
    #        Models
    # ---------------------
    args.duration = int((args.window.end - args.window.start) * args.preprocs.brain_resample_rate)
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)
    # pretrained_weight = '../../results/20230429_sbj01_eegnet_cv_norm_regression/weights/model_last.pt'
    # brain_encoder.load_state_dict(torch.load(pretrained_weight))
    # print('resume from ', pretrained_weight)

    classifier = Classifier(args)

    # ---------------
    #      Loss
    # ---------------
    if args.criterion == 'clip':
        loss_func = CLIPLoss(args).to(device) # torch.nn.MSELoss(reduction="mean") #CLIPLoss(args).to(device)
        loss_func.train()
    elif args.criterion == 'mse':
        loss_func = torch.nn.MSELoss(reduction="mean")
    else:
        raise ValueError()
    # --------------------
    #      Optimizer
    # --------------------
    optimizer = torch.optim.Adam(
        list(brain_encoder.parameters()) + list(loss_func.parameters()), lr=float(args.lr),
    )

    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )
    elif args.lr_scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(m * args.epochs) for m in args.lr_multistep_mlstns],
            gamma=args.lr_step_gamma,
        )
    else:
        scheduler = None
    image_encoder.to(device)
    # ======================================
    best_acc = 0
    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        pbar.set_description("training {}/{} epoch".format(epoch, args.epochs))
        train_losses = []
        test_losses = []
        trainTop1accs = []
        trainTop10accs = []
        testTop1accs = []
        testTop10accs = []

        brain_encoder.train()
        pbar2 = tqdm(train_loader)
        for i, batch in enumerate(pbar2):

            # if len(batch) == 3:
            #     X, Y, subject_idxs = batch
            # elif len(batch) == 4:
            #     X, Y, subject_idxs, chunkIDs = batch
            #     assert (
            #         len(chunkIDs.unique()) == X.shape[0]
            #     ), "Duplicate segments in batch are not allowed. Aborting."
            # else:
            #     raise ValueError("Unexpected number of items from dataloader.")
            batch_eeg, batch_image = batch
            batch_eeg, batch_image = batch_eeg.to(device), batch_image.to(device)
            with torch.no_grad():
                Y = image_encoder.encode_image(batch_image).to(torch.float)
            X = batch_eeg
            # X = X - normalize_mean_X
            # X = X / normalize_std_X
            Y = Y - normalize_mean_Y
            Y = Y / normalize_std_Y
            
            # import pdb; pdb.set_trace()
            Z = brain_encoder(X, None)
            # import pdb; pdb.set_trace()
            loss = loss_func(Y, Z)
            with torch.no_grad():
                trainTop1acc, trainTop10acc = classifier(Z, Y)

            train_losses.append(loss.item())
            trainTop1accs.append(trainTop1acc)
            trainTop10accs.append(trainTop10acc)

            pbar.set_description("training {}/{} iters Train/Loss: {}, Train/Top1Acc: {}, Train/Top10Acc: {}".format(i, len(train_loader), loss.item(), trainTop1acc, trainTop10acc))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # if args.dataset == "Gwilliams2022":
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            # if args.dataset == "GOD":
            #     optimizer.zero_grad()
            #     loss.backward()
            #     optimizer.step()
            #     # get_grad(brain_encoder)
            # break

        # Accumulate gradients for Gwilliams for the whole epoch
        # if args.dataset == "Brennan2018":
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        brain_encoder.eval()
        for batch in test_loader:

            with torch.no_grad():

                # if len(batch) == 3:
                #     X, Y, subject_idxs = batch
                # elif len(batch) == 4:
                #     X, Y, subject_idxs, chunkIDs = batch
                # else:
                #     raise ValueError("Unexpected number of items from dataloader.")

                # X, Y = X.to(device), Y.to(device)
                batch_eeg, batch_image = batch
                batch_eeg, batch_image = batch_eeg.to(device), batch_image.to(device)
                with torch.no_grad():
                    Y = image_encoder.encode_image(batch_image).to(torch.float)
                X = batch_eeg.to(torch.float)
                # X = X - normalize_mean_X
                # X = X / normalize_std_X
                Y = Y - normalize_mean_Y
                Y = Y / normalize_std_Y
                Z = brain_encoder(X, None)  # 0.96 GB

                loss = loss_func(Y, Z)

                testTop1acc, testTop10acc = classifier(Z, Y, test=True)  # ( 250, 1024, 360 )

            test_losses.append(loss.item())
            testTop1accs.append(testTop1acc)
            testTop10accs.append(testTop10acc)
        temp = loss_func.temp.item() if isinstance(loss_func, CLIPLoss) else 0.0
        print(
            f"Ep {epoch}/{args.epochs} | ",
            f"train l: {np.mean(train_losses):.3f} | ",
            f"test l: {np.mean(test_losses):.3f} | ",
            f"trainTop10acc: {np.mean(trainTop10accs):.3f} | ",
            f"testTop10acc: {np.mean(testTop10accs):.3f} | ",
            f"lr: {optimizer.param_groups[0]['lr']:.5f}",
            f"temp: {temp:.3f}",
        )
        pkl_logger.log({
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "trainTop1acc": np.mean(trainTop1accs),
                "trainTop10acc": np.mean(trainTop10accs),
                "testTop1acc": np.mean(testTop1accs),
                "testTop10acc": np.mean(testTop10accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item() if isinstance(loss_func, CLIPLoss) else 0,
            }, 'logs')

        if usewandb:
            performance_now = {
                "epoch": epoch,
                "train_loss": np.mean(train_losses),
                "test_loss": np.mean(test_losses),
                "trainTop1acc": np.mean(trainTop1accs),
                "trainTop10acc": np.mean(trainTop10accs),
                "testTop1acc": np.mean(testTop1accs),
                "testTop10acc": np.mean(testTop10accs),
                "lrate": optimizer.param_groups[0]["lr"],
                "temp": loss_func.temp.item() if isinstance(loss_func, CLIPLoss) else 0,
            }
            wandb.log(performance_now)

        if scheduler is not None:
            scheduler.step()

        savedir = os.path.join(args.save_root, 'weights')
        last_weight_file = os.path.join(savedir, "model_last.pt")
        torch.save(brain_encoder.state_dict(), last_weight_file)
        print('model is saved as ', last_weight_file)
        if best_acc < np.mean(testTop10accs):
            best_weight_file = os.path.join(savedir, "model_best.pt")
            torch.save(brain_encoder.state_dict(), best_weight_file)
            best_acc =  np.mean(testTop10accs)
            print('best model is updated !!, {}'.format(best_acc), best_weight_file)

if __name__ == "__main__":
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="meg_ssl/task_configs/"):
        args = compose(config_name='regression_eegnet_deep_clip_things5')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    run(args)