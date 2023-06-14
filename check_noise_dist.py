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
import os
from torch.utils.data import DataLoader
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import stats

import random

def run(args):
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
    source_dataset = GODDatasetBase(args, 'train', return_label=True)
    outlier_dataset = GODDatasetBase(args, 'val', return_label=True,
                                        mean_X= source_dataset.mean_X,
                                        mean_Y=source_dataset.mean_Y,
                                        std_X=source_dataset.std_X,
                                        std_Y=source_dataset.std_Y
                                    )
    ind_tr = list(range(0, 3000)) + list(range(3600, 6600)) #+ list(range(7200, 21600)) # + list(range(7200, 13200)) + list(range(14400, 20400))
    ind_te = list(range(3000,3600)) + list(range(6600, 7200)) # + list(range(13200, 14400)) + list(range(20400, 21600))
    train_dataset = Subset(source_dataset, ind_tr)
    val_dataset   = Subset(source_dataset, ind_te)
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
    brain_encoder = get_model(args).to(device) #BrainEncoder(args).to(device)

    weight_dir = os.path.join(os.path.join('/',*args.save_root.split('/')[:-1]), 'weights')
    last_weight_file = os.path.join(weight_dir, "model_last.pt")
    best_weight_file = os.path.join(weight_dir, "model_best.pt")
    if os.path.exists(best_weight_file):
        brain_encoder.load_state_dict(torch.load(best_weight_file))
        print('weight is loaded from ', best_weight_file)
    else:
        brain_encoder.load_state_dict(torch.load(last_weight_file))
        print('weight is loaded from ', last_weight_file)


    classifier = Classifier(args)
    
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

            testTop1acc, testTop10acc = classifier(Z, Y, test=True)  # ( 250, 1024, 360 )

    Zs = torch.cat(Zs, dim=0)
    Ys = torch.cat(Ys, dim=0)
    Ls = torch.cat(Ls, dim=0).detach().cpu().numpy()
    raw_Ys = Ys
    raw_Zs = Zs
    raw_Es = Zs-Ys
    
    Zs = Zs - Zs.mean(dim=0, keepdims=True)
    Zs = Zs / Zs.std(dim=0, keepdims=True)
    Zs = Zs - Zs.mean(dim=1, keepdims=True)
    Zs = Zs / Zs.std(dim=1, keepdims=True)
    Ys = Ys - Ys.mean(dim=1, keepdims=True)
    Ys = Ys / Ys.std(dim=1, keepdims=True)
    
    normalized_Es  = Zs-Ys
    
    raw_Es = raw_Es.detach().cpu().numpy()
    raw_gaussian_param = norm.fit(raw_Es)
    raw_x = np.linspace(np.min(raw_Es),np.max(raw_Es),100)
    raw_pdf_fitted = norm.pdf(raw_x, loc=raw_gaussian_param[0], scale=raw_gaussian_param[1])
    # raw_pdf_fitted *= raw_Es.size
    normalized_Es = normalized_Es.detach().cpu().numpy()
    normalized_gaussian_param = norm.fit(normalized_Es)
    normalized_x = np.linspace(np.min(normalized_Es),np.max(normalized_Es),100)
    normalized_pdf_fitted = norm.pdf(normalized_x, loc=normalized_gaussian_param[0], scale=normalized_gaussian_param[1])
    # normalized_pdf_fitted *= normalized_Es.size
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 12))
    weights = np.ones(raw_Es.size)/float(raw_Es.size)
    axes[0,0].hist(raw_Es.flatten(), bins=30, weights=weights)
    axes[0,0].plot(raw_x, raw_pdf_fitted, 'r-')
    axes[0,0].set_title('raw_noise-mean: {:.3f}-std: {:.3f}'.format(*raw_gaussian_param))
    axes[1,0].hist(raw_Zs.detach().cpu().numpy().flatten(), bins=50, color='b', alpha=0.3, label='Z')
    # axes[1,0].hist(raw_Ys.detach().cpu().numpy().flatten(), bins=50, color='r', alpha=0.3, label='Y')
    axes[1,0].legend()
    weights = np.ones(normalized_Es.size)/float(normalized_Es.size)
    axes[0,1].hist(normalized_Es.flatten(), bins=30, weights=weights)
    axes[0,1].set_title('normalize_noise-mean: {:.3f}-std: {:.3f}'.format(*normalized_gaussian_param))
    axes[0,1].plot(normalized_x, normalized_pdf_fitted, 'r-')
    axes[1,1].hist(Zs.detach().cpu().numpy().flatten(), bins=50, color='b', alpha=0.3, label='Z')
    axes[1,1].hist(Ys.detach().cpu().numpy().flatten(), bins=50, color='r', alpha=0.3, label='Y')
    axes[1,1].legend()
    plt.savefig(os.path.join(args.save_root, 'noise_dist.png'))
    ### 歪度の検定
    skewness, skew_p_value = stats.skewtest(raw_Es.flatten())
    print(f'Skewness test statistic (raw): {skewness}, p-value: {skew_p_value}')
    skewness, skew_p_value = stats.skewtest(normalized_Es.flatten())
    print(f'Skewness test statistic (normalized): {skewness}, p-value: {skew_p_value}')

    ### 尖度の検定
    kurtosis, kurt_p_value = stats.kurtosistest(raw_Es.flatten())
    print(f'Kurtosis test statistic (raw): {kurtosis}, p-value: {kurt_p_value}')
    kurtosis, kurt_p_value = stats.kurtosistest(normalized_Es.flatten())
    print(f'kurtosis test statistic (normalized): {kurtosis}, p-value: {kurt_p_value}')

    # コルモゴロフ=スミルノフ検定
    print('Korgormov-smirnof test (raw): ',stats.ks_1samp(raw_Es.flatten(), stats.norm.cdf))
    print('Korgormov-smirnof test (normalized): ',stats.ks_1samp(normalized_Es.flatten(), stats.norm.cdf))
    
if __name__ == '__main__':
    from hydra import initialize, compose
    with initialize(version_base=None, config_path="../configs/"):
        args = compose(config_name='20230429_sbj01_eegnet_regression')
    args.save_root = os.path.join(args.save_root, 'noise_dist')
    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)
    run(args)