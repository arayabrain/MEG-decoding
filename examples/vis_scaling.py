import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
import os

RESULT_ROOT = '/home/yainoue/meg2image/results_ssl'


VC_fs1000_bp2_120_dur208_patch4_dirs = {
    1000: 'scmbm_4-fs1000-dura200-1k',
    5000: 'scmbm_4-fs1000-dura200-5k',
    10000: 'scmbm_4-fs1000-dura200-10k',
    31138: 'scmbm_4-fs1000-dura200',
    'name': 'VC_fs1000_bp2_120_dur208_patch4'
}

VC_fs1000_bp2_120_dur512_patch4_dirs = {
    1000: 'scmbm_4-fs1000-dura500-1k',
    5000: 'scmbm_4-fs1000-dura500-5k',
    10000: 'scmbm_4-fs1000-dura500-10k',
    31138: 'scmbm_4-fs1000-dura500',
    'name': 'VC_fs1000_bp2_120_dur512_patch4'
}

VC_fs1000_bp2_120_dur208_patch16_dirs = {
    1000: 'scmbm_16-fs1000-dura200-1k',
    5000: 'scmbm_16-fs1000-dura200-5k',
    10000: 'scmbm_16-fs1000-dura200-10k',
    31138: 'scmbm_16-fs1000-dura200',
    'name': 'VC_fs1000_bp2_120_dur208_patch16'
}

def get_log_file(dirname:str):
    dirname = os.path.join(RESULT_ROOT, dirname, 'log')
    filelist = []
    for f in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, f)):
            filelist.append(f)
    print(dirname, '\n', filelist)
    assert len(filelist) == 1

    return os.path.join(RESULT_ROOT, dirname, filelist[-1])

def run(dirname_dict_list:list):
    log_list = []
    fig, axes = plt.subplots(ncols=2, figsize=(10, 5))
    for dirname_dict in dirname_dict_list:
        val_loss_logs = {}
        val_corr_logs = {}
        val_loss = []
        val_corr = []
        data_size = []
        for key, value in dirname_dict.items():
            if key == 'name':
                # continue
                val_loss_logs[key] =  value
                val_corr_logs[key] = value
                name = value
            else:
                filepath = get_log_file(value)
                with open(filepath, 'rb') as f:
                    log_data = pickle.load(f)
                train_logs = log_data['train']
                val_logs = log_data['val']
                min_idx = np.argmin(val_logs['val_loss'])
                val_loss_logs[key] = val_logs['val_loss'][min_idx]
                val_corr_logs[key] = val_logs['val_corr'][min_idx]
                data_size.append(key)
                val_loss.append(val_logs['val_loss'][min_idx])
                val_corr.append(val_logs['val_corr'][min_idx])
        axes[0].plot(data_size, val_loss, '-o', label=name)
        axes[1].plot(data_size, val_corr, '-o', label=name)
    axes[0].set_xlabel('data size')
    axes[1].set_xlabel('data size')
    axes[0].set_ylabel('reconstruction loss')
    axes[1].set_ylabel('reconstruction cor')
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    axes[0].legend()
    axes[1].legend()

    savefile = 'tmps/scaling/VC.png'
    plt.savefig(savefile)
    print('save to ', savefile)
        



if __name__ == '__main__':
    dict_list = [
        VC_fs1000_bp2_120_dur208_patch4_dirs,
        VC_fs1000_bp2_120_dur512_patch4_dirs,
        VC_fs1000_bp2_120_dur208_patch16_dirs
    ]
    run(dict_list)

