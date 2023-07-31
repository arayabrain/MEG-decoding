import argparse
from hydra import compose, initialize
import sys
sys.path.append('.')
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models import get_model_and_trainer
from omegaconf import OmegaConf
import wandb

# get dataset
def get_dataset(cfg:OmegaConf):
    dataset_names:dict = cfg.dataset
    dataset_yamls:dict = cfg.dataset_yaml
    num_trial_limit:dict = cfg.total_limit
    preproc_config:OmegaConf = cfg.preprocess
    h5_root:str = cfg.h5_root
    image_preprocs:list = []
    meg_preprocs:list = []
    only_meg:bool = True
    on_memory:bool = False
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit, 
                                      h5_root, image_preprocs, meg_preprocs, only_meg, on_memory)
    
    return dataset_dict['train'], dataset_dict['val']
    

# get model
def get_model(config, usewandb, devuce_count):
    model, trainer = get_model_and_trainer(config, device_count=devuce_count, usewandb=usewandb, only_model=False)
    return model. trainer



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
        wandb.init(project='ssl-meg-movie')        
    else:
        usewandb=False
    print('============================ settings ==============================')
    print(OmegaConf.to_yaml(cfg))
    print('=============================== END ================================')
    # dataset
    train_dataset, val_dataset = parse_dataset(cfg)

    # model
    model, trainer = get_model_and_trainer(cfg, usewandb, device_counts)

    trainer.fit(model, train_dataset, val_dataset, ckpt_path=cfg.resume_path)


# argparse
def parse_args():
    parser = argparse.ArgumentParser(
                        prog='Self-Supervised learning for MEG-image decoding',
                        description='with Yanagisawa-Lab',
                        epilog='Created by inoue@araya.org')
    parser.add_argument('--config', type=str, default='ssl/ssl_configs/test_config.yaml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--wandbkey', type=str, default=None) # default='/home/yainoue/wandb_inoue.txt')
    parser.add_argument('--device_counts', type=int, default=1)

    args = parser.parse_args()
    print('args: \n', args)
    return args


if __name__ == '__main__':
    args = parse_args()
    initialize(config_path='meg_ssl/ssl_configs/')
    cfg = compose(args.config)
    cfg.resume_path = args.resume

    run(cfg, args.wandbkey, args.device_counts)