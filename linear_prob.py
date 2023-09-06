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
from cuml.linear_model import LogisticRegression as LogisticRegressionCUDA
from cuml.svm import SVC as SVC_CUDA
import cuml
import cupy as cp
from tqdm import tqdm
import time

# get MEG encoder
def get_meg_encoder(config, usewandb, device_count, meg_exp):
    model, _ = get_model_and_trainer(config, device_count=device_count, usewandb=usewandb, only_model=True)
    assert not isinstance(model, tuple)
    if meg_exp == 'random':
        pass
    else:
        model.load_state_dict(torch.load(config.meg_encoder_path))
        print('model weight is load from ', config.meg_encoder_path)
    return model

class LinearProbDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        # self.img_transform = img_transform
        # self.vit_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


    def __len__(self):
        return len(self.dataset)

    @property
    def datasets(self):
        return self.dataset.datasets

    def __getitem__(self, idx):
        ret = {}
        eeg, image, label = self.dataset[idx]
        # image_raw = numpy2image(image)
        # image_raw = self.vit_processor(images=image_raw, return_tensors="pt")
        # image_raw['pixel_values'] = image_raw['pixel_values'].squeeze(0)
        # ret['image_raw'] = image_raw
        # ret['image'] = self.img_transform(image.astype(np.float32)/255.0)
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
    ret_image_label:bool = True
    dataset_dict:dict = parse_dataset(dataset_names, dataset_yamls, preproc_config, num_trial_limit,
                                    h5_root, image_preprocs, meg_preprocs, only_meg, on_memory, ret_image_label)
    # import pdb; pdb.set_trace()
    # crop_pix = int(cfg.training.crop_ratio*cfg.training.img_size)
    # img_transform_train = transforms.Compose([
    #     normalize,

    #     transforms.Resize((512, 512)),
    #     random_crop(cfg.training.img_size-crop_pix, p=0.5),

    #     transforms.Resize((512, 512)),
    #     channel_last
    # ])
    # img_transform_test = transforms.Compose([
    #     normalize,

    #     transforms.Resize((512, 512)),
    #     channel_last
    # ])
    return LinearProbDataset(dataset_dict['train']), LinearProbDataset(dataset_dict['val'])


# get meg embeddings
def get_meg_embeddings(model, dataset, device, reduction='none', output='latent'):
    model.to(device)
    model.eval()
    dataloader = DataLoader(dataset, batch_size=100, shuffle=False, num_workers=4, collate_fn=None)
    meg = []
    image_label = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            image_label += data['label'].cpu().numpy().tolist()
            if output=='raw':
                meg_embeddings = data['eeg']
            elif output=='latent':
                meg_embeddings, mask, ids_restore = model.forward_encoder(data['eeg'].to(device), mask_ratio=-1)
            elif output=='reconst':
                loss, meg_embeddings, mask = model(data['eeg'].to(device), mask_ratio=0)
                # print(loss)
            meg.append(meg_embeddings.cpu().numpy())

    meg = np.concatenate(meg, axis=0) # (6000, 53, 1024)
    if reduction == 'none':
        meg = np.reshape(meg, (len(meg), -1))
    elif reduction == 'mean':
        meg = np.mean(meg, axis=-1)
    elif reduction == 'max':
        meg = np.max(meg, axis=-1)
    return meg, np.array(image_label)

# get decoder

# fit linear model
def fit(model:str, train_x, train_y, test_x, test_y):
    if model == 'linear':
        clf = LogisticRegression(penalty='l1', solver='saga')
        print('logistic regression')
    elif model == 'linear_cuda':
        clf = LogisticRegressionCUDA(penalty='l1')
        print('logistic regression CUDA')
    elif model == 'svm_cuda':
        clf = cuml.MBSGDClassifier(loss='hinge', penalty='l1') # SVC_CUDA(kernel='poly', degree=2, gamma='auto', C=1)
        print('linear support vector machine CUDA')
    elif model == 'knn_cuda':
        clf = cuml.KNeighborsClassifier(n_neighbors=10)
        print('Nearest Neighbor CUDA')
    else:
        raise NotImplementedError
    if 'cuda' in model:
        train_x = cp.asarray(train_x)
        train_y = cp.asarray(train_y)
        test_x = cp.asarray(test_x)
        test_y = cp.asarray(test_y)
    print('fitting start')
    clf.fit(train_x, train_y)
    if 'cuda' in model:
        train_pred = clf.predict(train_x)
        train_score = np.mean(train_pred==train_y)
        test_pred = clf.predict(test_x)
        test_score = np.mean(test_pred==test_y)
        
    else:
        train_score = clf.score(train_x, train_y)
        test_score = clf.score(test_x, test_y)
    print('train: ', train_score)
    print('test: ', test_score)
    return clf, train_score, test_score



def count_object_class(dataset):
    counts = {k:0 for k in range(150)}
    for data in dataset:
        label = data['label']
        c = int(np.floor(label/8))
        counts[c] += 1
    return counts

def main(cfg, meg_exp):
    # set seed
    set_seed(42)
    # get dataset
    train_dataset, val_dataset = get_dataset(cfg)
    # val_counts = count_object_class(val_dataset)
    # train_counts = count_object_class(train_dataset)
    # print(val_counts)
    # print(train_counts)
    cfg.meg_encoder.parameters.in_chans = val_dataset.datasets[0].num_electrodes
    cfg.meg_encoder.parameters.time_len = int(np.floor(cfg.preprocess.meg_duration * cfg.preprocess.brain_resample_rate))
    # get encoder
    encoder = get_meg_encoder(cfg, usewandb=False, device_count=1, meg_exp=meg_exp)
    # get embeddings
    train_meg, train_label = get_meg_embeddings(encoder, train_dataset, device='cuda')
    val_meg, val_label = get_meg_embeddings(encoder, val_dataset, device='cuda')
    # fit linear model
    model = cfg.prob_model
    # clf, train_score, test_score = fit(model, train_meg, train_label, val_meg, val_label)
    clf2, train_score2, test_score2 = fit(model, train_meg, (np.floor(train_label/8)).astype(np.int32), val_meg, (np.floor(val_label/8).astype(np.int32)))
    # train_score = 0
    # test_score=0
    # train_score2=0
    # test_score2 = 0
    # save model
    save_path = os.path.join(cfg.save_path, model)
    os.makedirs(save_path, exist_ok=True)
    # save results
    savefilename = os.path.join(save_path, 'results.txt')
    with open(savefilename, 'a') as f:
        f.write(f'{meg_exp}\n')
        # f.write('train (1/1200): {}\n'.format(train_score))
        # f.write('test (1/1200): {}\n'.format(test_score))
        f.write('train (1/150): {}\n'.format(train_score2))
        f.write('test (1/150): {}\n'.format(test_score2))
    print('result is saved as ', savefilename)


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
    print(meg_cfg)
    meg_exp_list = ['scmbm_4-fs1000-dura200', 'scmbm_4-fs1000-dura200-10k', 'scmbm_4-fs1000-dura200-5k', 
                    'scmbm_4-fs1000-dura200-2.5k', 'scmbm_4-fs1000-dura200-1k', 'random']
    meg_encoder_path_pattern = meg_cfg.meg_encoder_path
    for meg_exp in meg_exp_list:
        meg_cfg.meg_encoder_path = meg_encoder_path_pattern.format(meg_exp=meg_exp)
        p1 = time.time()
        main(meg_cfg, meg_exp)
        print('duration: {} sec'.format(time.time()-p1))