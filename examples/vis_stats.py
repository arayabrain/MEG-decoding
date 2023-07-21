import pandas as pd
import matplotlib.pyplot as plt
from bdpy.stats import corrmat

filename1 = '/home/yainoue/meg2image/codes/MEG-decoding/results/GenericObjectDecoding/stats.csv'
filename2 = '/home/yainoue/meg2image/codes/MEG-decoding/results/GenericObjectDecoding/stats_meg.csv'
feat_name = ['CLIP-ViTB-16', '1st layer of MatConvNet', '3rd layer of MatConvNet', '5th layer of MatConvNet', '8th layer of MatConvNet']
data = pd.read_csv(filename1)
data_meg = pd.read_csv(filename2)
target_column = ['top10acc', 'acc', 'mean_corr']
feat_gb = data.groupby('feat')
fig,axes = plt.subplots(ncols=3, figsize=(15,5))
# import pdb; pdb.set_trace()
mean = feat_gb[target_column].agg('mean')
std = feat_gb[target_column].agg('std')
xs = []
for i, k in enumerate(feat_gb.groups.keys()):
    g = feat_gb.get_group(k)
    k = feat_name[i]
    axes[0].plot([k]*len(g), g['acc'], 'bo-')
    axes[1].plot([k]*len(g), g['mean_corr'], 'bo-')
    axes[2].plot([k]*len(g), g['top10acc'], 'bo-')
    xs.append(k)

# xs = feat_name
# import pdb; pdb.set_trace()
axes[0].plot(xs, mean['acc'], 'b-', label='fMRI')
axes[1].plot(xs, mean['mean_corr'], 'b-', label='fMRI')
axes[2].plot(xs, mean['top10acc'], 'b-', label='fMRI')
feat_gb_meg = data_meg.groupby('feat')
# import pdb; pdb.set_trace()
mean = feat_gb_meg[target_column].agg('mean')
std = feat_gb_meg[target_column].agg('std')
xs = []
for i, k in enumerate(feat_gb.groups.keys()):
    g = feat_gb_meg.get_group(k)
    k = feat_name[i]
    axes[0].plot([k]*len(g), g['acc'], 'mo-')
    axes[1].plot([k]*len(g), g['mean_corr'], 'mo-')
    axes[2].plot([k]*len(g), g['top10acc'], 'mo-')
    xs.append(k)
# xs = feat_name
# import pdb; pdb.set_trace()
axes[0].plot(xs, mean['acc'], 'm-', label='MEG')
axes[1].plot(xs, mean['mean_corr'], 'm-', label='MEG')
axes[2].plot(xs, mean['top10acc'], 'm-', label='MEG')
axes[0].set_ylim(0,1)
axes[1].set_ylim(0,1)
axes[2].set_ylim(0,1)
axes[0].legend()
axes[1].legend()
axes[2].legend()
axes[0].set_ylabel('seen identification acc')
axes[0].xaxis.set_tick_params(rotation=90)
# axes[0].set_xticklabels(axes[0].get_xticks(), label=feat_name, rotation = 90)
# axes[0].set_title('seen identification accuracy')
axes[1].set_title('mean correlation')
axes[2].set_title('top10 acc (/50)')
plt.savefig(filename1.replace('.csv', '.png'), bbox_inches = "tight")
plt.close()
print('save ', filename1.replace('.csv', '.png'))
