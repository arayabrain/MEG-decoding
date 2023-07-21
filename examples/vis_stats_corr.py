import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

file_list =  ['/home/yainoue/meg2image/results/20230429_sbj01_eegnet_cv_norm_regression/corr_per_dim_image.pkl',
              '/home/yainoue/meg2image/results/20230621_sbj01_eegnet_cv_norm_regression_cnn1/corr_per_dim_image.pkl',
              '/home/yainoue/meg2image/results/20230623_sbj01_eegnet_cv_norm_regression_cnn3/corr_per_dim_image.pkl',
              '/home/yainoue/meg2image/results/20230622_sbj01_eegnet_cv_norm_regression_cnn5/corr_per_dim_image.pkl',
              '/home/yainoue/meg2image/results/20230622_sbj01_eegnet_cv_norm_regression_cnn8/corr_per_dim_image.pkl']
feats = ['clip', 'cnn1', 'cnn3', 'cnn5', 'cnn8']
save_root = 'results/GenericObjectDecoding'
fig, axes = plt.subplots(ncols=2, figsize=(12,5))
for feat, file in zip(feats, file_list):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    axes[0].plot(np.arange(len(data['corr_dim'])), data['corr_dim'], 'o', label=feat)
    axes[0].set_xlabel('dim')
    axes[0].set_ylabel('corr')
    print('mean of corr_dim', np.mean(data['corr_dim']))

    axes[1].plot(np.arange(len(data['corr_image'])), data['corr_image'], 'o', label=feat)
    axes[1].set_xlabel('image_id')
    axes[1].set_ylabel('corr')
    axes[0].legend()
    axes[1].legend()
    print('mean of corr_image', np.mean(data['corr_image']))
    savefile = os.path.join(save_root, 'MEG-corr_per_dim_image.png')
    plt.savefig(savefile)