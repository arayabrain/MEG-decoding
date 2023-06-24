import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import os

loss_key = ['train_loss', 'test_loss']
acc_key = ['trainTop1acc', 'trainTop10acc', 'testTop1acc', 'testTop10acc']
lr_key = ['lrate']
temp_key = ['temp']

def get_data(pkl_file:str):
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def visualize(logs, savefile):
    fig, axes = plt.subplots(ncols=4, figsize=(32,8))
    epochs = np.arange(len(logs))
    ax=axes[0]
    for k in loss_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')   
    ax.legend()
    ax=axes[1]
    for k in acc_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend()
    ax=axes[2]
    for k in lr_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('lr') 
    ax.legend()
    ax=axes[3]
    for k in temp_key:
        ax.plot(epochs, logs[k].to_list(), label=k)
    ax.set_xlabel('epoch')
    ax.set_ylabel('temp') 
    ax.legend()
    plt.savefig(savefile)
    print('save as ', savefile)

def parse_data(filepath):
    data = get_data(filepath)
    log_names = list(data.keys())
    savedir = os.path.dirname(filepath)
    for name in log_names:
        logs = data[name]
        columns = logs.keys()
        logs = pd.DataFrame(logs)
        savefile = os.path.join(savedir, f'{name}_learning_curve.png')
        visualize(logs, savefile)
        print('==================={}==============='.format(name))
        print(logs)
        for i, row in logs.iterrows():
            print(row.to_list())


if __name__ == '__main__':
    # filepath = '/home/yainoue/meg2image/results/20230417_sbj01_seq2stat/runs/2023-04-19 11:08:38.266128'
    # filepath = '/home/yainoue/meg2image/results/20230419_sbj01_seq2stat/runs/2023-04-20 01:39:30.648046'
    # filepath = '/home/yainoue/meg2image/results/20230419_sbj01_seq2stat2/runs/2023-04-20 13:06:53.825175'
    # filepath = '/home/yainoue/meg2image/results/20230420ß_sbj01_linear/runs/2023-04-20 22:04:01.525056'
    # filepath =  '/home/yainoue/meg2image/results/20230420_sbj01_seq2stat2/runs/2023-04-21 02:04:47.889601'
    # filepath = '/home/yainoue/meg2image/results/20230423_sbj010203_seq2stats_regression/runs/2023-04-24 00:01:31.293371'
    # filepath = '/home/yainoue/meg2image/results/20230424_sbj01_seq2stat/runs/2023-04-25 00:34:28.269896'
    # filepath = '/home/yainoue/meg2image/results/20230425_sbj01_seq2stat_cv/runs/2023-04-25 21:00:55.483213'
    # filepath = '/home/yainoue/meg2image/results/20230425_sbj01_seq2stat_cv_norm_wo_dilation/runs/2023-04-26 11:27:01.295948'
    # filepath = '/home/yainoue/meg2image/results/20230426_all_seq2stat_cv_norm_wo_dilation/runs/2023-04-26 16:05:36.849896'
    # filepath = '/home/yainoue/meg2image/results/20230426_all_seq2stat_cv_norm_wo_dilation/runs/2023-04-26 16:19:27.018492'
    # filepath = '/home/yainoue/meg2image/results/20230413_sbj01/runs/2023-04-13 21:52:17.333087'#'home/yainoue/meg2image/results/test/2023-04-11 18:06:44.273986'
    # filepath = '/home/yainoue/meg2image/results/20230413_sbj01_seq2stat/runs/2023-04-16 19:21:02.983480'
    # filepath = '/home/yainoue/meg2image/results/20230412_mini/runs/2023-04-13 15:17:53.310665'
    # filepath = '/home/yainoue/meg2image/results/20230427_sbj01_eegnet_cv_norm/runs/2023-04-28 03:53:04.866741' 
    # filepath = '/home/yainoue/meg2image/results/20230428_sbj01_eegnet_cv_norm/runs/2023-04-28 21:05:42.739003'
    # filepath = '/home/yainoue/meg2image/results/20230429_sbj01_eegnet_cv_norm_regression/runs/2023-04-30 04:16:13.736655'
    filepath = '/home/yainoue/meg2image/results/20230501_all_eegnet_cv_norm_regression/runs/2023-05-01 17:01:17.163323'
    # filepath = '/home/yainoue/meg2image/results/20230515_sbj02_eegnet_cv_norm_regression/runs/2023-05-15 01:08:45.283225'
    # filepath = '/home/yainoue/meg2image/results/20230517_all_eegnet_cv_norm_regression_cogitat/runs/2023-05-17 05:57:53.604625'
    # filepath = '/home/yainoue/meg2image/results/20230518_all_eegnet_cv_norm_regression/runs/2023-05-17 18:50:02.743497'
    filepath = '/home/yainoue/meg2image/results/20230519_all_eegnet_cv_norm_regression_src_reconst/runs/2023-05-19 23:13:54.848307'
    filepath = '/home/yainoue/meg2image/results/20230522_sbj01_eegnet_cv_norm_regression_src_reconst/runs/2023-05-23 19:29:43.093427'
    filepath = '/home/yainoue/meg2image/results/20230524_all_eegnet_cv_norm_regression_src_reconst/runs/2023-05-24 18:33:34.519060'
    filepath = '/home/yainoue/meg2image/results/20230531_sbj02_eegnet_cv_norm_regression_src_reconst/runs/2023-06-01 00:20:53.904804'
    filepath = '/home/yainoue/meg2image/results/20230601_sbj03_eegnet_cv_norm_regression_src_reconst/runs/2023-06-01 21:04:03.763788'
    filepath = '/home/yainoue/meg2image/results/20230606_sbj02_eegnet_cv_norm/runs/2023-06-06 23:58:40.026704'
    filepath = '/home/yainoue/meg2image/results/20230621_sbj01_eegnet_cv_norm_regression_cnn1/runs/2023-06-21 13:36:57.211277',
    filepath = '/home/yainoue/meg2image/results/20230622_sbj01_eegnet_cv_norm_regression_cnn5/runs/2023-06-22 07:08:03.174032'
    filepath = '/home/yainoue/meg2image/results/20230622_sbj01_eegnet_cv_norm_regression_cnn8/runs/2023-06-22 23:43:31.666899'
    filepath = '/home/yainoue/meg2image/results/20230623_sbj01_eegnet_cv_norm_regression_cnn3/runs/2023-06-23 16:58:15.761800'
    parse_data(filepath)
