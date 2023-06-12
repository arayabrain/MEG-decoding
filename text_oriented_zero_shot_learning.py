import os, sys, random
import numpy as np
import torch
from tqdm import tqdm, trange
from termcolor import cprint
# import wandb
import pandas as pd
import json

from constants import device
# from speech_decoding.dataclass.brennan2018 import Brennan2018Dataset
# from speech_decoding.dataclass.gwilliams2022 import (
#     Gwilliams2022SentenceSplit,
#     Gwilliams2022ShallowSplit,
#     Gwilliams2022DeepSplit,
#     Gwilliams2022Collator,
# )
from torch.utils.data import DataLoader
try:
    # import cv2
    from omegaconf import DictConfig, open_dict
    from meg_decoding.models import get_model, Classifier
    from meg_decoding.utils.get_dataloaders import get_dataloaders, get_samplers
    from meg_decoding.utils.loss import *
    from meg_decoding.dataclass.god import GODDatasetBase, GODCollator
    from meg_decoding.utils.loggers import Pickleogger
    # from meg_decoding.clip_utils.get_embedding import get_language_model
    from torch.utils.data.dataset import Subset
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pickle
    from PIL import Image
except ModuleNotFoundError :
    pass

def get_language_model(prompt_dict:dict, savedir):
    if os.path.exists(os.path.join(savedir, 'text_features')):
        
        text_features = torch.load(os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'r') as f:
            prompts = f.readlines()
    else:
        import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model = model.eval()
        prompts = []
        prefix = prompt_dict['prefix']
        for i, t in prompt_dict.items():
            if i == 'prefix':
                continue
            prompts.append(t+'\n')
        text = clip.tokenize([prefix + s.replace('\n','') for s in prompts]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        # with open(os.path.join(savedir, 'text_features'), 'wb') as f:
        torch.save(text_features, os.path.join(savedir, 'text_features'))
        with open(os.path.join(savedir, 'prompts.txt'), 'w') as f:
            f.writelines(prompts)
    return text_features, prompts

def run(args, prompts:list, text_embeddings:list, beta:float) -> None:

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

    cprint(f"Current working directory : {os.getcwd()}")
    cprint(args, color="white")

    # -----------------------
    #       Dataloader
    # -----------------------
    if args.dataset == "GOD":
        source_dataset = GODDatasetBase(args, 'train', return_label=True)
        outlier_dataset = GODDatasetBase(args, 'val', return_label=True,
                                         mean_X= source_dataset.mean_X,
                                         mean_Y=source_dataset.mean_Y,
                                         std_X=source_dataset.std_X,
                                         std_Y=source_dataset.std_Y
                                        )
        with open("/home/yainoue/meg2image/codes/MEG-decoding/data/ImageNet/val_features.pkl", "rb") as f:
            imagenet_data = pickle.load(f)
            imagenet_Y = np.zeros((len(imagenet_data), 512))
            imagenet_name = [None] * len(imagenet_data)
            cnt = 0
            for k, v in imagenet_data.items():
                imagenet_Y[cnt] = v # v: 512
                imagenet_name[cnt] = k
                cnt += 1
            imagenet_Y -= source_dataset.mean_Y
            imagenet_Y /= source_dataset.std_Y
            imagenet_Y = torch.Tensor(imagenet_Y).to(device)

        # train_size = int(np.round(len(source_dataset)*0.8))
        # val_size = len(source_dataset) - train_size

        # train_dataset, val_dataset = torch.utils.data.random_split(source_dataset, [train_size, val_size])
        ind_tr = list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
        ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
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
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )
            test_loader = DataLoader(
                # val_dataset, #
                outlier_dataset,  # val_dataset
                batch_size=50, # args.batch_size,
                drop_last=True,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                worker_init_fn=seed_worker,
                generator=g,
            )

    else:
        raise ValueError("Unknown dataset")

    if args.use_wandb:
        wandb.config = {k: v for k, v in dict(args).items() if k not in ["root_dir", "wandb"]}
        wandb.init(
            project=args.wandb.project,
            entity=args.wandb.entity,
            config=wandb.config,
            save_code=True,
        )
        wandb.run.name = args.wandb.run_name + "_" + args.split_mode
        wandb.run.save()

    # ---------------------
    #        Models
    # ---------------------
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(os.path.join('/',*args.save_root.split('/')[:-2]), 'weights')
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
    # train_losses = []
    test_losses = []
    testTop1accs = []
    testTop10accs = []

    Zs = []
    Ys = []
    Ls = []
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
            Zs.append(Z)
            Ys.append(Y)
            Ls.append(Labels)

            loss = loss_func(Y, Z)

            testTop1acc, testTop10acc = classifier(Z, Y, test=True)  # ( 250, 1024, 360 )

        test_losses.append(loss.item())
        testTop1accs.append(testTop1acc)
        testTop10accs.append(testTop10acc)
    Zs = torch.cat(Zs, dim=0)
    Ys = torch.cat(Ys, dim=0)
    Ls = torch.cat(Ls, dim=0).detach().cpu().numpy()

    print(
        # f"train l: {np.mean(train_losses):.3f} | ",
        f"test l: {np.mean(test_losses):.3f} | ",
        # f"trainTop10acc: {np.mean(trainTop10accs):.3f} | ",
        f"testTop10acc: {np.mean(testTop10accs):.3f} | ",
        # f"temp: {loss_func.temp.item():.3f}",
    )


    # 仮説1:判定に偏りがある。-> あるサンプルのimageの特徴量がMEGの潜在空間ににているかどうかを判定するだけの基準になっているのではないか？
    Zs = Zs - Zs.mean(dim=0, keepdims=True)
    Zs = Zs / Zs.std(dim=0, keepdims=True)
    Zs = Zs - Zs.mean(dim=1, keepdims=True)
    Zs = Zs / Zs.std(dim=1, keepdims=True)

    # prompt orienting
    T = text_embeddings.cpu().numpy()
    T = T - source_dataset.mean_Y
    T = T / source_dataset.std_Y
    T = torch.Tensor(T).to(device)
    Zs_oriented = Zs + beta* T
    Ys_with_imagenet = torch.cat([Ys, imagenet_Y], dim=0)
    similarity = calc_similarity(Zs_oriented, Ys_with_imagenet)
    top5_similarity = {'query_image_id':[],
                       'top1_image_id':[], 'top2_image_id':[], 'top3_image_id':[], 'top4_image_id':[], 'top5_image_id':[]}


    for i, l in enumerate(Ls):
        sim_vec = similarity[i,:]
        top5_similarity['query_image_id'].append(l)
        ranking = np.argsort(sim_vec)[::-1][:5] + 1 # 1始まりにする
        for k in range(1,6):
            key = f'top{k}_image_id'
            if ranking[k-1] <= 50:
                image_name = str(ranking[k-1])
            else:
                image_name = imagenet_name[ranking[k-1]-50-1]
            top5_similarity[key].append(image_name)
    top5_similarity = pd.DataFrame(top5_similarity)
    top5_similarity.to_csv(os.path.join(args.save_root, 'top5_with_imagenet_val_beta{}.csv'.format(beta)))



def save_top5_prediction(args, beta):
    top5_similarity = pd.read_csv(os.path.join(args.save_root, 'top5_with_imagenet_val_beta{}.csv'.format(beta)))
    split = 5
    unit = int(len(top5_similarity) / split)
    imagenet_val_root = '/storage/dataset/image/ImageNet/ILSVRC2012_val/'
    for i in range(split):
        image_tiles = []
        for j in range(i*unit, (i+1)*unit):
            row = top5_similarity.iloc[j]
            row_image_list = []
            for key in ['top1_image_id', 'top2_image_id', 'top3_image_id', 'top4_image_id', 'top5_image_id']:
                image_file_name = os.path.join(imagenet_val_root, str(row[key]))
                if os.path.exists(image_file_name):
                    image = Image.open(image_file_name)
                    image = image.resize((112,112))
                    image = np.array(image)
                    assert image.shape[0] == 112, 'image has shape {}'.format(image.shape)
                else:
                    image = np.ones([112,112,3]).astype(np.uint8)
                if image.ndim == 2:
                    image = np.stack([image]*3, axis=-1)
                row_image_list.append(image)
            row_image = np.concatenate(row_image_list, axis=0)
            image_tiles.append(row_image)
            # import pdb; pdb.set_trace()
        image_tiles = np.concatenate(image_tiles, axis=1)
        pil_img = Image.fromarray(image_tiles)
        pil_img.save(os.path.join(args.save_root, f'top5_with_imagenet_val-{i}-beta{beta}.png'))
        # cv2.write_image(os.path.join(args.save_root, f'top5_with_imagenet_val-beta{beta}-{i}.png'), image_tiles)
        print('saved as ', os.path.join(args.save_root, f'top5_with_imagenet_val-{i}-beta{beta}.png'))

def boxplot_and_plot(bp_array, plot_array_label, ax):
    # array: n_image x dims(512)
    plot_array = bp_array[plot_array_label]
    ax.boxplot(bp_array)
    for l, ar in zip(plot_array_label, plot_array):
        ax.plot(np.arange(len(ar)), ar, label=str(l))
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_xlabel('unit id')
    ax.set_ylabel('logits')

def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = torch.empty(batch_size, gt_size).to('cuda')
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = (x[i] @ y[j]) / max((x[i].norm() * y[j].norm()), 1e-8)
    return similarity.cpu().numpy()


if __name__ == "__main__":
    prompt_root = '/home/yainoue/meg2image/codes/MEG-decoding/data/prompts'
    prompt_sub_dir = 'person'
    prompt_dir = os.path.join(prompt_root, prompt_sub_dir)
    prompt_dict_file = os.path.join(prompt_dir, 'classification1.json')
    with open(prompt_dict_file, 'r') as f:
        prompt_dict = json.load(f)
    text_features, prompts =  get_language_model(prompt_dict, prompt_dir)
    print('subdir: ', prompt_dir)
    print('use prompt: ', prompts)
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230429_sbj01_eegnet_regression')
    if not os.path.exists(os.path.join(args.save_root, 'weights')):
        os.makedirs(os.path.join(args.save_root, 'weights'))
    args.save_root = os.path.join(args.save_root, 'text_oriented', prompt_sub_dir)
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    for beta in [0, 0.01, 0.02, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0, 10]:
        # beta = 1.0
        run(args, prompts, text_features, beta=beta) 
        save_top5_prediction(args, beta)
    
