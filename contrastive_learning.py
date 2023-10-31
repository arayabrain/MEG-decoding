import argparse
from hydra import compose, initialize
import sys
sys.path.append('.')
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models import get_model_and_trainer
from meg_ssl.models.image_encoder import get_image_encoder
from meg_ssl.models.decoder import get_decoder
from meg_ssl.utils import set_seed
from meg_ssl.trainers.contrastive_trainer import AlignTrainer

from omegaconf import OmegaConf
import wandb
import numpy as np
import functools
import os
from meg_ssl.utils.image_preprocess import numpy2image, transform2vit_image_only_inputs
import torch

# get dataset
def get_dataset(cfg:OmegaConf):
    dataset_names:dict = cfg.dataset_name
    # import pdb; pdb.set_trace()
    dataset_yamls:dict = cfg.dataset_yaml
    num_trial_limit:dict = cfg.total_limit
    preproc_config:OmegaConf = cfg.preprocess
    h5_root:str = cfg.h5_root
    image_preprocs:list = []# [numpy2image]
    meg_preprocs:list = []
    only_meg:bool = False # with image
    on_memory:bool = False
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit,
                                      h5_root, image_preprocs, meg_preprocs, only_meg, on_memory)

    return dataset_dict['train'], dataset_dict['val']


# get model
def get_model(config, device_count, usewandb):
    meg_encoder, _ = get_model_and_trainer(config, device_count=device_count, usewandb=usewandb, only_model=True)
    image_encoder, image_processor = get_image_encoder(config.image_encoder.name, config.image_encoder.parameters)
    decoder = get_decoder(config.decoder.name, config.decoder.parameters)
    # TODO:
    # import pdb; pdb.set_trace()
    return meg_encoder, image_encoder, image_processor, decoder

def get_contrastive_trainer(config, device_count, usewandb):
    training_config = config.training
    return AlignTrainer(
        training_config, device_count, usewandb
    )


# run
def run(cfg:OmegaConf, wandb_key_path:str, device_counts:int):
    # wandb setting
    if wandb_key_path is not None:
        usewandb=True
        with open(wandb_key_path, 'r') as f:
            wandb_key = f.read()
            wandb_key = wandb_key.split('\n')[0]
        wandb.login(key = f'{wandb_key}')
        # wandb.init(project='llm_hackason-'+args.config)
        # wandb.init(project='llm_hackason-new_prompt')
        wandb.init(project='meg-god')
    else:
        usewandb=False


    cfg.meg_encoder.parameters.time_len = int(np.floor(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate))
    print('============================ settings ==============================')
    print(OmegaConf.to_yaml(cfg))
    print('=============================== END ================================')
    # dataset
    train_dataset, val_dataset = get_dataset(cfg)
    cfg.meg_encoder.parameters.in_chans = train_dataset.datasets[0].num_electrodes
    print('num_electrodes: ', cfg.meg_encoder.parameters.in_chans)
    print('num trials: Train: {}, Val: {}'.format(len(train_dataset), len(val_dataset)))
    # model
    meg_encoder, image_encoder, image_processor, decoder = get_model(cfg, device_count=device_counts, usewandb=usewandb) # get_model(cfg, usewandb, device_counts)
    trainer = get_contrastive_trainer(cfg, device_counts, usewandb)

    def collate_fn(batch):
        # create new batch
        batch_meg = torch.stack([torch.from_numpy(b[0]) for b in batch])
        # batch_image = [numpy2image(b[1]) for b in batch]
        # batch_image = transform2vit_image_only_inputs(batch_image)
        # batch_image = image_processor(batch_image['text'], batch_image['images'], return_tensors="pt", padding=True)
        batch_image = torch.stack([image_processor(numpy2image(b[1])) for b in batch])
        return batch_meg, batch_image
    # image preprocess function
    image_processing_funcs = []
    trainer.fit(meg_encoder, image_encoder, decoder, image_processing_funcs, train_dataset, val_dataset,
                collate_fn=collate_fn,
            meg_encoder_ckpt_path=cfg.meg_encoder_path,
            image_encoder_ckpt_path=cfg.image_encoder_path,
            decoder_ckpt_path=cfg.resume_path)


# argparse
def parse_args():
    parser = argparse.ArgumentParser(
                        prog='down-stream task for MEG-image decoding',
                        description='with Yanagisawa-Lab',
                        epilog='Created by inoue@araya.org')
    parser.add_argument('--config', type=str, default='ssl/ssl_configs/test_config.yaml')
    parser.add_argument('--meg_model', type=str, default=None)
    parser.add_argument('--vision_model', type=str, default='vit_clip')
    parser.add_argument('--decode_model', type=str, default='mlp')
    parser.add_argument('--preprocess', type=str, default=None)
    parser.add_argument('--pretrained_ckpt', type=str, default=None)
    parser.add_argument('--wandbkey', type=str, default=None) # default='/home/yainoue/wandb_inoue.txt')
    parser.add_argument('--device_counts', type=int, default=1)
    parser.add_argument('--exp', type=str, default=None)
    parser.add_argument('--h5name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    print('args: \n', args)
    return args


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    with initialize(config_path='meg_ssl/task_configs/'):
        cfg = compose(args.config)
    with initialize(config_path='meg_ssl/ssl_configs/model'):
        cfg.meg_encoder = compose(args.meg_model)
    print('INFO ========= model config is overrided by ', args.meg_model)
    with initialize(config_path='meg_ssl/ssl_configs/preprocess'):
        cfg.preprocess = compose(args.preprocess)
    print('INFO ========= preprocess config is overrided by', args.preprocess)
    with initialize(config_path='meg_ssl/task_configs/model'):
        cfg.image_encoder = compose(args.vision_model)
    print('INFO ========= image_encoder config is overrided by', args.vision_model)
    with initialize(config_path='meg_ssl/task_configs/model'):
        cfg.decoder = compose(args.decode_model)
    print('INFO ========= decoder config is overrided by', args.decode_model)
    # import pdb; pdb.set_trace()
    # num_electrodes, fs, bpがh5ファイルに関係している
    if args.h5name is None:
        cfg.h5_root = cfg.h5_root.format(h5_name='fs{}-bp{}_{}'.format(cfg.preprocess.brain_resample_rate, *cfg.preprocess.bandpass_filter))
    else:
        cfg.h5_root = cfg.h5_root.format(h5_name=args.h5name)
    if args.exp is None:
        args.exp = args.config
    cfg.training.logdir = cfg.training.logdir.format(exp_name=args.exp)
    cfg.training.ckpt_dir = cfg.training.ckpt_dir.format(exp_name=args.exp)
    if cfg.resume_path is None:
        cfg.resume_path = args.resume
    cfg.meg_encoder_path = cfg.meg_encoder_path.format(exp_name=args.exp)

    # decoderのinput features は前段のモデルのパラメタに依存する & output featuresはimage encoderの埋め込み次元に依存する
    if cfg.decoder.name == 'mlp':
        if cfg.training.global_pool:
            cfg.decoder.parameters.input_features = cfg.meg_encoder.parameters.embed_dim
        else:
            cfg.decoder.parameters.input_features = int(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate / cfg.meg_encoder.parameters.patch_size * cfg.meg_encoder.parameters.embed_dim) # time_len * resample_rate / patch_size * num_dim
        cfg.decoder.parameters.output_features =  512 # image_encoder.vision_embed_dim
    if cfg.decoder.name == 'mapping':
        cfg.decoder.parameters.input_features = 1024  # 52 x 1024
        cfg.decoder.parameters.middle_features = int(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate / cfg.meg_encoder.parameters.patch_size) # time_len * resample_rate / patch_size * num_dim
        cfg.decoder.parameters.output_features =  512
        # print(cfg.decoder.parameters.middle_features)
        # import pdb; pdb.set_trace()
    run(cfg, args.wandbkey, args.device_counts)