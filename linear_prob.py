import os, sys
import numpy as np
import torch
import argparse
from meg_ssl.utils.commons import set_seed
from hydra import compose, initialize
from meg_ssl.models import get_model_and_trainer
from meg_ssl.dataclass import parse_dataset
from meg_ssl.models.dc_ldm.ldm_for_eeg import eLDM
from meg_ssl.utils.image_preprocess import numpy2image
from transformers import AutoProcessor
import torchvision.transforms as transforms
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# get MEG encoder
def get_meg_encoder(config, usewandb, device_count):
    model = get_model_and_trainer(config, device_count=device_count, usewandb=usewandb, only_model=True)
    assert not isinstance(model, tuple)
    model.load_state_dict(torch.load(config.meg_encoder_path))
    return model

class LinearProbDataset():
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
        eeg, image, label = self.dataset[idx]
        image_raw = numpy2image(image)
        image_raw = self.vit_processor(images=image_raw, return_tensors="pt")
        image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        ret['image_raw'] = image_raw
        ret['image'] = self.img_transform(image.astype(np.float32)/255.0)
        ret['eeg'] = torch.from_numpy(eeg)
        ret['label'] = label # dummy

        return ret

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
    return LinearProbDataset(dataset_dict['train'], img_transform_train), LinearProbDataset(dataset_dict['val'], img_transform_test)


# get meg embeddings
def get_meg_embeddings(model, dataset, device):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4, collate_fn=None)
    meg = []
    image_label = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            image_label += data['label'].cpu().numpy().tolist()
            meg_embeddings = model(data['eeg'].to(device))
            meg.append(meg_embeddings.cpu().numpy())

    meg = np.concatenate(meg, axis=0)
    return meg, np.array(image_label)

# get decoder

# fit linear model
def fit(model:str, train_x, train_y, test_x, test_y):
    if model == 'linear':
        clf = LogisticRegression()
    else:
        raise NotImplementedError
    clf.fit(train_x, train_y)
    train_score = clf.score(train_x, train_y)
    test_score = clf.score(test_x, test_y)
    print('train: ', train_score)
    print('test: ', test_score)
    return clf, train_score, test_score


def main(cfg):
    # set seed
    set_seed(42)
    # get dataset
    train_dataset, val_dataset = get_dataset(cfg)
    # get encoder
    encoder = get_meg_encoder(cfg, usewandb=False, device_count=1)
    # get embeddings
    train_meg, train_label = get_meg_embeddings(encoder, train_dataset, device='cuda')
    val_meg, val_label = get_meg_embeddings(encoder, val_dataset, device='cuda')
    # fit linear model
    model = cfg.prob_model
    clf, train_score, test_score = fit(model, train_meg, train_label, val_meg, val_label)
    # save model
    save_path = os.path.join(cfg.save_path, cfg.model)
    os.makedirs(save_path, exist_ok=True)
    # save results
    with open(os.path.join(save_path, 'results.txt'), 'w') as f:
        f.write('train: {}\n'.format(train_score))
        f.write('test: {}\n'.format(test_score))
    # save config
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        f.write(cfg.pretty())


if __name__ == '__main__':
    with initialize(config_path='meg_ssl/task_configs/'):
        meg_cfg = compose('linear_prob')
    with initialize(config_path='meg_ssl/ssl_configs/model'):
        meg_cfg.meg_encoder = compose(meg_cfg.meg_encoder)

    with initialize(config_path='meg_ssl/ssl_configs/preprocess'):
        meg_cfg.preprocess = compose(meg_cfg.preprocess)
    # num_electrodes, fs, bpがh5ファイルに関係している
    meg_cfg.h5_root = meg_cfg.h5_root.format(h5_name='fs{}-bp{}_{}'.format(meg_cfg.preprocess.brain_resample_rate, *meg_cfg.preprocess.bandpass_filter))

    exp_name = 'scmbm_4-fs1000-dura200' + '-' + 'linear_prob'
    print('experiment name is ', exp_name)
    main(meg_cfg)