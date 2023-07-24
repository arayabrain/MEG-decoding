import matplotlib.pyplot as plt
import numpy as np
import os
from twonn_torch.two_nn_utils import twonn_dimension
from twonn_torch.visualizations import vis_different_scale
from typing import Tuple
import pickle
from tqdm import tqdm
import torch
import random

def get_feature_paths(data_root):
        image_paths = []
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.npy'):
                    image_paths.append(os.path.join(root, file))
        return image_paths

def get_dataset(data_root:str)->Tuple:
    # file_names = os.listdir(data_root)
    file_names = get_feature_paths(data_root)
    file_names.sort()
    data_dict = {}
    for filename in file_names:
        filepath = os.path.join(data_root, filename)
        if ('l_train.npy' == filename) or ('l_val.npy' == filename) or ('l_test.npy' == filename) or ('y_' in filename):
            print('skip {}'.format(filename))
        elif '.npy' in filename:
            data = np.load(filepath)
            # idx = random.sample(range(len(data)), 50)
            if 'train' in filename:
                data = data[:1200]
            if 'test' in filename:
                data = data[:50]
            # import pdb; pdb.set_trace()
            key = filename.replace('.npy', '')
            key = '-'.join(key.split('/')[-2:])
            data_dict[key] = data
        else:
            print('skip {}'.format(filename))
    return data_dict

def predict_id(data_dict:dict)->Tuple:
    ids = {}
    N_dims = {}
    for name, data in tqdm(data_dict.items()):
        if data.ndim != 2:
            data = data.reshape(data.shape[0], -1)
        if torch.cuda.is_available():
            data = torch.from_numpy(data).to('cuda')
        print('processing {}...shape:{}'.format(name, data.shape))
        N_dims[name] = data.shape[1]
        ids[name] = twonn_dimension(data,return_xy=False)

    return ids, N_dims

def run(datadir:str, feature_names=None):
    id_path = os.path.join(datadir, 'ids.pkl')
    if os.path.exists(id_path):
        with open(id_path, 'rb')as f:
            data = pickle.load(f)
            id_dict, N_dims_dict = data['ids'], data['N_dims']
    else:
        data_dict = get_dataset(datadir)
        id_dict, N_dims_dict = predict_id(data_dict)
        with open(id_path, 'wb')as f:
            pickle.dump({'ids': id_dict, 'N_dims': N_dims_dict}, f)

    if feature_names is None:
        layer_names = list(id_dict.keys())
        ids = list(id_dict.values())
        n_dims = list(N_dims_dict.values())
    else:
        layer_names = []
        ids = []
        n_dims = []
        for name in feature_names:
            layer_names.append(name)
            ids.append(id_dict[name])
            n_dims.append(N_dims_dict[name])

    # plt.plot(layer_names, ids, '-o', label='Intrinsic Dimension')
    # plt.plot(layer_names, n_dims, '--o', label='Feature Dimension')
    savepath = os.path.join(datadir, 'ids.png')
    vis_different_scale(layer_names, ids, n_dims, savepath)


if __name__ == '__main__':
    # from hydra import initialize, compose
    # with initialize(version_base=None, config_path="../configs/"):
    #     args = compose(config_name='20230429_sbj01_eegnet_regression')
    # datadir = os.path.join(args.save_root, 'features')
    datadir = '/home/yainoue/meg2image/results/20230721_intrinsic_dimension' 
    run(datadir, feature_names=None)