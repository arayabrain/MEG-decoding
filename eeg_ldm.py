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


def wandb_init(config, output_path):
    # wandb.init( project='dreamdiffusion',
    #             group="stageB_dc-ldm",
    #             anonymous="allow",
    #             config=config,
    #             reinit=True)
    create_readme(config, output_path)

def wandb_finish():
    wandb.finish()

def to_image(img):
    if img.shape[-1] != 3:
        img = rearrange(img, 'c h w -> h w c')
    img = 255. * img
    return Image.fromarray(img.astype(np.uint8))

def channel_last(img):
        if img.shape[-1] == 3:
            return img
        return rearrange(img, 'c h w -> h w c')

def get_eval_metric(samples, avg=True):
    metric_list = ['mse', 'pcc', 'ssim', 'psm']
    res_list = []

    gt_images = [img[0] for img in samples]
    gt_images = rearrange(np.stack(gt_images), 'n c h w -> n h w c')
    samples_to_run = np.arange(1, len(samples[0])) if avg else [1]
    for m in metric_list:
        res_part = []
        for s in samples_to_run:
            pred_images = [img[s] for img in samples]
            pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
            res = get_similarity_metric(pred_images, gt_images, method='pair-wise', metric_name=m)
            res_part.append(np.mean(res))
        res_list.append(np.mean(res_part))
    res_part = []
    for s in samples_to_run:
        pred_images = [img[s] for img in samples]
        pred_images = rearrange(np.stack(pred_images), 'n c h w -> n h w c')
        res = get_similarity_metric(pred_images, gt_images, 'class', None,
                        n_way=50, num_trials=50, top_k=1, device='cuda')
        res_part.append(np.mean(res))
    res_list.append(np.mean(res_part))
    res_list.append(np.max(res_part))
    metric_list.append('top-1-class')
    metric_list.append('top-1-class (max)')
    return res_list, metric_list

def generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config):
    grid, _ = generative_model.generate(eeg_latents_dataset_train, config.num_samples,
                config.ddim_steps, config.HW, 10) # generate 10 instances
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path, 'samples_train.png'))
    # wandb.log({'summary/samples_train': wandb.Image(grid_imgs)})

    grid, samples = generative_model.generate(eeg_latents_dataset_test, config.num_samples,
                config.ddim_steps, config.HW)
    grid_imgs = Image.fromarray(grid.astype(np.uint8))
    grid_imgs.save(os.path.join(config.output_path,f'./samples_test.png'))
    for sp_idx, imgs in enumerate(samples):
        for copy_idx, img in enumerate(imgs[1:]):
            img = rearrange(img, 'c h w -> h w c')
            Image.fromarray(img).save(os.path.join(config.output_path,
                            f'./test{sp_idx}-{copy_idx}.png'))

    # wandb.log({f'summary/samples_test': wandb.Image(grid_imgs)})

    metric, metric_list = get_eval_metric(samples, avg=config.eval_avg)
    metric_dict = {f'summary/pair-wise_{k}':v for k, v in zip(metric_list[:-2], metric[:-2])}
    metric_dict[f'summary/{metric_list[-2]}'] = metric[-2]
    metric_dict[f'summary/{metric_list[-1]}'] = metric[-1]
    # wandb.log(metric_dict)

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

def main(config, args):
    # project setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    crop_pix = int(config.crop_ratio*config.img_size)
    img_transform_train = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        random_crop(config.img_size-crop_pix, p=0.5),

        transforms.Resize((512, 512)),
        channel_last
    ])
    img_transform_test = transforms.Compose([
        normalize,

        transforms.Resize((512, 512)),
        channel_last
    ])
    if config.dataset == 'EEG':

        eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset(eeg_signals_path = config.eeg_signals_path, splits_path = config.splits_path,
                image_transform=[img_transform_train, img_transform_test], subject = config.subject)
        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_viz( image_transform=[img_transform_train, img_transform_test])
        num_voxels = eeg_latents_dataset_train.data_len
        # prepare pretrained mbm

        pretrain_mbm_metafile = torch.load(config.pretrain_mbm_path, map_location='cpu')
        collate_fn = None

    elif config.dataset == 'MEG':
        from hydra import compose, initialize
        from meg_ssl.dataclass import parse_dataset
        # TODO: read config and get dataset
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
        from transformers import AutoProcessor
        from meg_ssl.utils.image_preprocess import numpy2image

        class DiffusionDataset():
            def __init__(self, dataset, split='train'):
                self.dataset = dataset
                if split=='train':
                    self.img_transform = img_transform_train
                else:
                    self.img_transform = img_transform_test
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

            return DiffusionDataset(dataset_dict['train'], 'train'), DiffusionDataset(dataset_dict['val'], 'val')
        eeg_latents_dataset_train, eeg_latents_dataset_test = get_dataset(meg_cfg)
        # eeg_latents_dataset_train, eeg_latents_dataset_test = create_EEG_dataset_viz( image_transform=[img_transform_train, img_transform_test])
        num_voxels = int(meg_cfg.preprocess.meg_duration * meg_cfg.preprocess.brain_resample_rate) # eeg_latents_dataset_train.datasets[0].num_electrodes
        meg_encoder_pretrained_path = os.path.join(meg_cfg.meg_encoder_path.format(exp_name=args.meg_exp))
        meg_cfg.meg_encoder.parameters.in_chans = eeg_latents_dataset_test.datasets[0].num_electrodes
        pretrain_mbm_metafile = {
            'model':torch.load(meg_encoder_pretrained_path),
            'config': meg_cfg.meg_encoder.parameters
        }
        # img_transform_train, img_transform_test

        def train_collate_fn(batch):
            new_batch = {}
            image_raw = [numpy2image(b[1]) for b in batch]
            image_raw = vit_processor(images=image_raw, return_tensors="pt")
            image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
            new_batch['image_raw'] = image_raw
            new_batch['image'] = torch.stack([img_transform_train(b[1]/255.0) for b in batch])
            new_batch['eeg'] = torch.stack([torch.from_numpy(b[0]) for b in batch])
            new_batch['label'] = torch.empty(len(new_batch['image']), 1) # dummy

            return new_batch
        def test_collate_fn(batch):
            new_batch = {}
            image_raw = [numpy2image(b[1]) for b in batch]
            image_raw = vit_processor(images=image_raw, return_tensors="pt")
            image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
            new_batch['image_raw'] = image_raw
            new_batch['image'] = torch.stack([img_transform_test(b[1]/255.0) for b in batch])
            new_batch['eeg'] = torch.stack([torch.from_numpy(b[0]) for b in batch])
            new_batch['label'] = torch.empty(len(new_batch['image']), 1) # dummy

            return new_batch

    else:
        raise NotImplementedError




    # print(num_voxels)


    # create generateive model
    generative_model = eLDM(pretrain_mbm_metafile, num_voxels,
                device=device, pretrain_root=config.pretrain_gm_path, logger=config.logger,
                ddim_steps=config.ddim_steps, global_pool=config.global_pool, use_time_cond=config.use_time_cond, clip_tune = config.clip_tune, cls_tune = config.cls_tune)

    # resume training if applicable
    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        generative_model.model.load_state_dict(model_meta['model_state_dict'])
        print('model resumed')
    # finetune the model
    trainer = create_trainer(config.num_epoch, config.precision, config.accumulate_grad, config.logger, check_val_every_n_epoch=2)
    generative_model.finetune(trainer, eeg_latents_dataset_train, eeg_latents_dataset_test,
                config.batch_size, config.lr, config.output_path, config=config, collate_fn=None)

    # generate images
    # generate limited train images and generate images for subjects seperately
    # generate_images(generative_model, eeg_latents_dataset_train, eeg_latents_dataset_test, config)

    return

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

    # # distributed training parameters
    # parser.add_argument('--local_rank', type=int)

    return parser

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


def create_trainer(num_epoch, precision=32, accumulate_grad_batches=2,logger=None,check_val_every_n_epoch=0):
    acc = 'gpu' if torch.cuda.is_available() else 'cpu'
    return pl.Trainer(accelerator=acc, max_epochs=num_epoch, logger=logger,
            precision=precision, accumulate_grad_batches=accumulate_grad_batches,
            enable_checkpointing=False, enable_model_summary=False, # gradient_clip_val=0.5, # remove by inoue
            check_val_every_n_epoch=check_val_every_n_epoch)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    config = Config_Generative_Model()
    config = update_config(args, config)

    if config.checkpoint_path is not None:
        model_meta = torch.load(config.checkpoint_path, map_location='cpu')
        ckp = config.checkpoint_path
        config = model_meta['config']
        config.checkpoint_path = ckp
        print('Resuming from checkpoint: {}'.format(config.checkpoint_path))

    output_path = os.path.join(config.output_path, 'results', 'generation',  '%s'%(datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")))
    config.output_path = output_path
    os.makedirs(output_path, exist_ok=True)

    wandb_init(config, output_path)

    # logger = WandbLogger()
    config.logger = None # logger
    main(config, args)
