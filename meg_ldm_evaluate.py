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

def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()

def evaluate(Z, Y, index = None):
    # Z: (batch_size, 512)
    # Y: (gt_size, 512)
    binary_confusion_matrix = np.zeros([len(Z), len(Y)])
    similarity = calc_similarity(Z, Y)
    acc_tmp = np.zeros(len(similarity))
    # import pdb; pdb.set_trace()
    for i in range(len(similarity)):
        if index is None:
            index_ = i
        acc_tmp[i] = np.sum(similarity[i,:] < similarity[i,index_]) / (len(Y)-1)
        binary_confusion_matrix[i,similarity[i,:] < similarity[i,index_]] = 1 
        binary_confusion_matrix[i,similarity[i,:] > similarity[i,index_]] = -1 
    similarity_acc = np.mean(acc_tmp)
    

    print('Similarity Acc', similarity_acc)
    
    return similarity_acc, binary_confusion_matrix

# SETTINGS::
images_dir = '../../results_task/dream_diffusion/scmbm_4-fs1000-dura200-test_enc-6k/reconst/eval_val/'
num_trials = 1200
n_samples = 6
mode = 1 # 'avg' # 'third' # 'second' #2 # 'first' # 'avg' # first
tmp_save_dir = './tmps/ldm_eval/'
os.makedirs(tmp_save_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
model = model.eval()

gt_tmp_filepath = os.path.join(tmp_save_dir, 'gt.npy')
reconst_tmp_filepath = os.path.join(tmp_save_dir, 'reconst.npy')

if os.path.exists(gt_tmp_filepath):
    original_image_features = np.load(gt_tmp_filepath)
else:
    # gt
    original_image_features = []
    for i in tqdm(range(num_trials)):
        filename = os.path.join(images_dir, f'{i}-gt.png')
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
    for i in tqdm(range(num_trials)):
        trial_reconst_images = []
        for j in range(n_samples):
            filename = os.path.join(images_dir, f'{i}-{j}.png')
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
original_image_features = torch.from_numpy(original_image_features).to(torch.float32)
sic_list = []
top10_acc = []
top100_acc = []
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
        reconst_feats_avg = reconst_feats[:mode].mean(0)
        reconst_feats_avg = torch.from_numpy(reconst_feats_avg)
        acc, mat = evaluate(reconst_feats_avg.unsqueeze(0).to(torch.float32), original_image_features, index=i)

    top10_acc_flag = 1 if np.sum(mat==-1) < 10 else 0
    top100_acc_flag = 1 if np.sum(mat==-1) < 100 else 0
    sic_list.append(acc)
    top10_acc.append(top10_acc_flag)
    top100_acc.append(top100_acc_flag)

print(sic_list)
print('Seen Identification Acc:: ', np.mean(sic_list))
print('Top10 100 Acc: (10/1200)', np.mean(top10_acc), np.mean(top100_acc))



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