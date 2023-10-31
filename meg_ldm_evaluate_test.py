#from importlib.resources import path
import os
import torch
import clip
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
import pandas as pd
import time
from PIL import Image
from PIL import ImageFile  # 大きな画像もロード
import pickle
ImageFile.LOAD_TRUNCATED_IMAGES = True
import scipy.io
import random

def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()

def evaluate(Z, Y, index=None):
    # Z: (batch_size, 512)
    # Y: (gt_size, 512)
    binary_confusion_matrix = np.zeros([len(Z), len(Y)])
    similarity = calc_similarity(Z, Y)
    acc_tmp = np.zeros(len(similarity))
    # import pdb; pdb.set_trace()
    for i in range(len(similarity)):
        if index is None:
            index = i
        acc_tmp[i] = np.sum(similarity[i,:] < similarity[i,index]) / (len(Y)-1)
        binary_confusion_matrix[i,similarity[i,:] < similarity[i,index]] = 1 
        binary_confusion_matrix[i,similarity[i,:] > similarity[i,index]] = -1 
    # import pdb; pdb.set_trace()
    similarity_acc = np.mean(acc_tmp)
    

    print('Similarity Acc', similarity_acc)
    
    return similarity_acc, binary_confusion_matrix

# SETTINGS::
meg_label_dir = '/work/project/MEG_GOD/GOD_dataset/sbj01/labels/val_{}.mat'
meg_labels = []
for i in range(1,7):
    meg_label_path = meg_label_dir.format(str(i))
    data = scipy.io.loadmat(meg_label_path)
    meg_labels.append(data['vec_index'][0])
meg_labels = np.concatenate(meg_labels)
assert len(meg_labels) == 300
assert np.max(meg_labels) == 50

images_dir = '../../results_task/dream_diffusion/scmbm_4-fs1000-dura200-test_enc-6k/reconst/eval/'
num_trials = 50 #1200
n_samples = 3
mode = 1 # 'avg' # 'third' # 'second' #2 # 'first' # 'avg' # first
normalize_across_samples = True # False
with_train_image = False #True
tmp_save_dir = './tmps/ldm_eval/'
os.makedirs(tmp_save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.eval()

gt_tmp_filepath = os.path.join(tmp_save_dir, 'gt_test.npy')
reconst_tmp_filepath = os.path.join(tmp_save_dir, 'reconst_test.npy')

if os.path.exists(gt_tmp_filepath):
    original_image_features = np.load(gt_tmp_filepath)
    if with_train_image:
        train_image_features = np.load(gt_tmp_filepath.replace('_test', ''))
        original_image_features = np.concatenate([original_image_features, train_image_features], axis=0)
else:
    # gt
    original_image_features = []
    for i in tqdm(range(1, num_trials+1)):
        indices = np.where(meg_labels==i)[0]
        filename = os.path.join(images_dir, f'{indices[0]}-gt.png')
        image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        
        image_features = image_features.to('cpu').detach().numpy().copy()
        original_image_features.append(image_features)
    original_image_features = np.concatenate(original_image_features, axis=0)
    np.save(gt_tmp_filepath, original_image_features)

if os.path.exists(reconst_tmp_filepath):
    reconst_image_features = np.load(reconst_tmp_filepath)
else:
    # reconst
    reconst_image_features = []
    for i in tqdm(range(1, num_trials+1)):
        trial_reconst_images = []
        indices = np.where(meg_labels==i)[0]
        for ind in indices:
            for j in range(n_samples):
                filename = os.path.join(images_dir, f'{ind}-{j+3}.png')
                image = preprocess(Image.open(filename)).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model.encode_image(image)
                image_features = image_features.to('cpu').detach().numpy().copy()
                trial_reconst_images.append(image_features)
        trial_reconst_images = np.concatenate(trial_reconst_images, axis=0)
        reconst_image_features.append(trial_reconst_images)
    reconst_image_features = np.stack(reconst_image_features, axis=0)
    np.save(reconst_tmp_filepath, reconst_image_features)

print('original_image_features: ', original_image_features.shape)
print('reconst_image_featurs: ', reconst_image_features.shape)
if normalize_across_samples:
    print('normalize start')
    # gt
    original_image_features -= original_image_features.mean(0, keepdims=True)
    original_image_features /= original_image_features.std(0, keepdims=True)
    # pred
    reconst_image_features = reconst_image_features.reshape((-1, reconst_image_features.shape[-1]))
    reconst_image_features -= reconst_image_features.mean(0, keepdims=True)
    reconst_image_features /= reconst_image_features.std(0, keepdims=True)
    reconst_image_features = reconst_image_features.reshape(50, -1, reconst_image_features.shape[-1])


original_image_features = torch.from_numpy(original_image_features).to(torch.float32)
sic_list = []
top10_acc = []
top20_acc = []
for i in tqdm(range(num_trials)):
    reconst_feats = reconst_image_features[i] # n_samples x dims
    if mode == 'avg':
        reconst_feats_avg = reconst_feats.mean(0)
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)
    elif mode == 'first':
        reconst_feats_avg = reconst_feats[0]
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)
    elif mode == 'second':
        reconst_feats_avg = reconst_feats[1]
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)
    elif mode == 'third':
        reconst_feats_avg = reconst_feats[2]
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)
    
    elif isinstance(mode, int):
        indices = random.sample(list(np.arange(18)), mode)
        reconst_feats_avg = reconst_feats[:mode].mean(0)
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)
    # import pdb; pdb.set_trace()
    top10_acc_flag = 1 if np.sum(mat==-1) < 10 else 0
    top20_acc_flag = 1 if np.sum(mat==-1) < 20 else 0
    sic_list.append(acc)
    top10_acc.append(top10_acc_flag)
    top20_acc.append(top20_acc_flag)

print(sic_list)
print('Seen Identification Acc:: ', np.mean(sic_list))
print('Top10 20 Acc: (10/50)', np.mean(top10_acc), np.mean(top20_acc))



exit()

# images_dir = "./imagenet_fall2011_oneImagePerCat_21-2k_20230309/"
images_dir = "/storage/dataset/ECoG/internal/GODv2-4/images_trn/"
print('start getting image file name list')
list_images = os.listdir(images_dir)
print('end getting image file name list')
len(list_images)

image_names_list = []
image_feats_list = []

for img in tqdm(list_images):
    img_dir = images_dir + img
    image = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    
    image_features = image_features.to('cpu').detach().numpy().copy()
    
    image_feats_list.append(image_features)
    image_names_list.append(img)

image_names_np = np.array(image_names_list)
image_feats_np = np.array(image_feats_list)

image_feats_np = np.squeeze(image_feats_np)

name_feat_dict = {}
# for img_name  in image_names_list:
#     for feat_val in image_feats_list:
for img_name, feat_val in zip(image_names_list, image_feats_list):
    name_feat_dict[img_name] = np.squeeze(feat_val)

a_file = open("/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/vit-b16/train_features.pkl", "wb")
pickle.dump(name_feat_dict, a_file)
a_file.close()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/16", device=device)
# model = model.eval()

# images_dir = "./imagenet_fall2011_oneImagePerCat_21-2k_20230309/"
images_dir = "/storage/dataset/ECoG/internal/GODv2-4/images_val/"
print('start getting image file name list')
list_images = os.listdir(images_dir)
print('end getting image file name list')
len(list_images)

image_names_list = []
image_feats_list = []

for img in tqdm(list_images):
    img_dir = images_dir + img
    image = preprocess(Image.open(img_dir)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)

    
    image_features = image_features.to('cpu').detach().numpy().copy()
    
    image_feats_list.append(image_features)
    image_names_list.append(img)

image_names_np = np.array(image_names_list)
image_feats_np = np.array(image_feats_list)

image_feats_np = np.squeeze(image_feats_np)

name_feat_dict = {}
# for img_name  in image_names_list:
#     for feat_val in image_feats_list:
for img_name, feat_val in zip(image_names_list, image_feats_list):
    name_feat_dict[img_name] = np.squeeze(feat_val)

a_file = open("/home/yainoue/meg2image/codes/MEG-decoding/data/GOD/vit-b16/val_features.pkl", "wb")
pickle.dump(name_feat_dict, a_file)
a_file.close()