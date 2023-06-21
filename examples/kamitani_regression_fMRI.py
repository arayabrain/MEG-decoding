'''Generic Object Decoding: Feature prediction

Analysis summary
----------------

- Learning method:   Sparse linear regression
- Preprocessing:     Normalization and voxel selection
- Data:              GenericDecoding_demo
- Results format:    Pandas dataframe
'''


from __future__ import print_function

import os
import sys
import pickle
from itertools import product
from time import time

import numpy as np
import pandas as pd
from scipy import stats

from slir import SparseLinearRegression
from sklearn.linear_model import LinearRegression  # For quick demo

import bdpy
from bdpy.bdata import concat_dataset
from bdpy.ml import add_bias
from bdpy.preproc import select_top
from bdpy.stats import corrcoef
from bdpy.util import makedir_ifnot, get_refdata
from bdpy.dataform import append_dataframe
from bdpy.distcomp import DistComp



import os

class Config():
    analysis_name = 'GenericObjectDecoding'

    # Data settings
    # subjects = {'Subject1' : ['fMRI/data/Subject1.h5'], # top10 acc: /50 0.4 Similarity Acc 0.6591836734693877 (n_units 100)
    #             'Subject2' : ['fMRI/data/Subject2.h5'], # top10 acc: /50 0.46 Similarity Acc 0.6738775510204082
    #             'Subject3' : ['fMRI/data/Subject3.h5'],
    #             'Subject4' : ['fMRI/data/Subject4.h5'],
    #             'Subject5' : ['fMRI/data/Subject5.h5']}
    subjects = {'Subject1' : ['data/fMRI/Subject1.h5'], # top10 acc: /50 0.7 Similarity Acc 0.819 (n_units=1000, N=35)  top10 acc: /50 0.7 Similarity Acc 0.815 (n_units=1000, N=6)
                'Subject2' : ['data/fMRI/Subject2.h5'] # top10 acc: /50 0.64 Similarity Acc 0.796 (n_units=1000, N=35) top10 acc: /50 0.58 Similarity Acc  0.749 (n_units=1000, N=6)
                }

    rois = {'VC' : 'ROI_VC = 1'}#,
            # 'LVC' : 'ROI_LVC = 1',
            # 'HVC' : 'ROI_HVC = 1',
            # 'V1' : 'ROI_V1 = 1',
            # 'V2' : 'ROI_V2 = 1',
            # 'V3' : 'ROI_V3 = 1',
            # 'V4' : 'ROI_V4 = 1',
            # 'LOC' : 'ROI_LOC = 1',
            # 'FFA' : 'ROI_FFA = 1',
            # 'PPA' : 'ROI_PPA = 1'}

    num_voxel = {'VC' : 1000,
                'LVC' : 1000,
                'HVC' : 1000,
                'V1' : 500,
                'V2' : 500,
                'V3' : 500,
                'V4' : 500,
                'LOC' : 500,
                'FFA' : 500,
                'PPA' : 500}

    image_feature_file = 'data/fMRI/ImageFeatures.h5'
    # features = ['cnn1', 'cnn2', 'cnn3', 'cnn4', 'cnn5', 'cnn6', 'cnn7', 'cnn8', 'hmax1', 'hmax2', 'hmax3', 'gist', 'sift']
    features = ['cnn5']
    use_clip = True

    # Results settings
    results_dir = os.path.join('../results', analysis_name)
    results_file = os.path.join('../results', analysis_name + '.pkl')

    # Figure settings
    # roi_labels = ['V1', 'V2', 'V3', 'V4', 'LOC', 'FFA', 'PPA', 'LVC', 'HVC', 'VC']
    roi_labels = ['VC']


# Main #################################################################

def main():
    # Settings ---------------------------------------------------------
    config = Config()
    # Data settings
    subjects = config.subjects
    rois = config.rois
    num_voxel = config.num_voxel

    image_feature = config.image_feature_file
    features = config.features

    n_iter = 200

    results_dir = config.results_dir

    # Misc settings
    analysis_basename = os.path.basename(__file__)

    # Load data --------------------------------------------------------
    print('----------------------------------------')
    print('Loading data')

    data_all = {}
    for sbj in subjects:
        if len(subjects[sbj]) == 1:
            data_all[sbj] = bdpy.BData(subjects[sbj][0])
        else:
            # Concatenate data
            suc_cols = ['Run', 'Block']
            data_all[sbj] = concat_dataset([bdpy.BData(f) for f in subjects[sbj]],
                                           successive=suc_cols)

    data_feature = bdpy.BData(image_feature)

    # Add any additional processing to data here

    # Initialize directories -------------------------------------------
    makedir_ifnot(results_dir)
    makedir_ifnot('tmp')

    # Analysis loop ----------------------------------------------------
    print('----------------------------------------')
    print('Analysis loop')

    for sbj, roi, feat in product(subjects, rois, features):
        print('--------------------')
        print('Subject:    %s' % sbj)
        print('ROI:        %s' % roi)
        print('Num voxels: %d' % num_voxel[roi])
        print('Feature:    %s' % feat)

        # Distributed computation
        analysis_id = analysis_basename + '-' + sbj + '-' + roi + '-' + feat
        results_file = os.path.join(results_dir, analysis_id + '.pkl')

        if os.path.exists(results_file):
            print('%s is already done. Skipped.' % analysis_id)
            with open(results_file, 'rb') as f:
                results = pickle.load(f)
            Zs = results['predicted_feature_averaged'][0]
            Ys = results['true_feature_averaged'][0]
            # pred_y_pt = results['predicted_feature'][0][:300]
            # true_y_pt = results['true_feature'][0][:300]
            # test_label_pt = results['test_label'][0][:300]
            # Zs, Ys, Ls \
            #     = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
            print('Zs mean:', np.mean(Zs))
            evaluate(Zs, Ys)
            continue

        dist = DistComp(lockdir='tmp', comp_id=analysis_id)
        # if dist.islocked():
        #     print('%s is already running. Skipped.' % analysis_id)
        #     continue

        dist.lock()

        # Prepare data
        print('Preparing data')
        dat = data_all[sbj]

        x = dat.select(rois[roi])           # Brain data
        datatype = dat.select('DataType')   # Data type
        labels = dat.select('stimulus_id')  # Image labels in brain data

        y = data_feature.select(feat)             # Image features
        y_label = data_feature.select('ImageID')  # Image labels

        # For quick demo, reduce the number of units from 1000 to 100
        # y = y[:, :100]
        y = y[:, :1000]

        y_sorted = get_refdata(y, y_label, labels)  # Image features corresponding to brain data

        # Get training and test dataset
        i_train = (datatype == 1).flatten()    # Index for training
        i_test_pt = (datatype == 2).flatten()  # Index for perception test
        i_test_im = (datatype == 3).flatten()  # Index for imagery test
        i_test = i_test_pt + i_test_im

        x_train = x[i_train, :]
        x_test = x[i_test, :]

        y_train = y_sorted[i_train, :] # 1200
        y_test = y_sorted[i_test, :] # 1750+500
        if config.use_clip:
            import mat73
            train_features_path = '/work/project/MEG_GOD/GOD_dataset/clip_image_training.mat'
            test_features_path = '/work/project/MEG_GOD/GOD_dataset/clip_image_test.mat'
            train_data = mat73.loadmat(train_features_path)
            test_data = mat73.loadmat(test_features_path)
            # import pdb; pdb.set_trace()
            y = np.concatenate([train_data['vec'], test_data['vec']], axis=0)
            y_sorted = get_refdata(y, y_label, labels)
            y_train = y_sorted[i_train, :] # 1200
            y_test = y_sorted[i_test, :] # 1750+500

        # Feature prediction
        pred_y, true_y = feature_prediction(x_train, y_train,
                                            x_test, y_test,
                                            n_voxel=num_voxel[roi],
                                            n_iter=n_iter)

        # Separate results for perception and imagery tests
        i_pt = i_test_pt[i_test]  # Index for perception test within test
        i_im = i_test_im[i_test]  # Index for imagery test within test

        pred_y_pt = pred_y[i_pt, :]
        pred_y_im = pred_y[i_im, :]

        true_y_pt = true_y[i_pt, :]
        true_y_im = true_y[i_im, :]

        # Get averaged predicted feature
        test_label_pt = labels[i_test_pt, :].flatten()
        test_label_im = labels[i_test_im, :].flatten()

        pred_y_pt_av, true_y_pt_av, test_label_set_pt \
            = get_averaged_feature(pred_y_pt, true_y_pt, test_label_pt)
        pred_y_im_av, true_y_im_av, test_label_set_im \
            = get_averaged_feature(pred_y_im, true_y_im, test_label_im)

        # Get category averaged features
        catlabels_pt = np.vstack([int(n) for n in test_label_pt])  # Category labels (perception test)
        catlabels_im = np.vstack([int(n) for n in test_label_im])  # Category labels (imagery test)
        catlabels_set_pt = np.unique(catlabels_pt)                 # Category label set (perception test)
        catlabels_set_im = np.unique(catlabels_im)                 # Category label set (imagery test)

        y_catlabels = data_feature.select('CatID')   # Category labels in image features
        ind_catave = (data_feature.select('FeatureType') == 3).flatten()

        # y_catave_pt = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_pt)
        # y_catave_im = get_refdata(y[ind_catave, :], y_catlabels[ind_catave, :], catlabels_set_im)

        # Prepare result dataframe
        results = pd.DataFrame({'subject' : [sbj, sbj],
                                'roi' : [roi, roi],
                                'feature' : [feat, feat],
                                'test_type' : ['perception', 'imagery'],
                                'true_feature': [true_y_pt, true_y_im],
                                'predicted_feature': [pred_y_pt, pred_y_im], # 35回も同じものを見せている
                                'test_label' : [test_label_pt, test_label_im],
                                'test_label_set' : [test_label_set_pt, test_label_set_im],
                                'true_feature_averaged' : [true_y_pt_av, true_y_im_av],
                                'predicted_feature_averaged' : [pred_y_pt_av, pred_y_im_av],
                                'category_label_set' : [catlabels_set_pt, catlabels_set_im],
                                # 'category_feature_averaged' : [y_catave_pt, y_catave_im]
                                })

        # Save results
        makedir_ifnot(os.path.dirname(results_file))
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print('Saved %s' % results_file)

        dist.unlock()


# Functions ############################################################

def feature_prediction(x_train, y_train, x_test, y_test, n_voxel=500, n_iter=200):
    '''Run feature prediction

    Parameters
    ----------
    x_train, y_train : array_like [shape = (n_sample, n_voxel)]
        Brain data and image features for training
    x_test, y_test : array_like [shape = (n_sample, n_unit)]
        Brain data and image features for test
    n_voxel : int
        The number of voxels
    n_iter : int
        The number of iterations

    Returns
    -------
    predicted_label : array_like [shape = (n_sample, n_unit)]
        Predicted features
    ture_label : array_like [shape = (n_sample, n_unit)]
        True features in test data
    '''

    n_unit = y_train.shape[1]

    # Normalize brian data (x)
    norm_mean_x = np.mean(x_train, axis=0)
    norm_scale_x = np.std(x_train, axis=0, ddof=1)

    x_train = (x_train - norm_mean_x) / norm_scale_x
    x_test = (x_test - norm_mean_x) / norm_scale_x

    # Feature prediction for each unit
    print('Running feature prediction')

    y_true_list = []
    y_pred_list = []

    for i in range(n_unit):

        print('Unit %03d' % (i + 1))
        start_time = time()

        # Get unit features
        y_train_unit = y_train[:, i]
        y_test_unit =  y_test[:, i]

        # Normalize image features for training (y_train_unit)
        norm_mean_y = np.mean(y_train_unit, axis=0)
        std_y = np.std(y_train_unit, axis=0, ddof=1)
        norm_scale_y = 1 if std_y == 0 else std_y

        y_train_unit = (y_train_unit - norm_mean_y) / norm_scale_y

        # Voxel selection
        corr = corrcoef(y_train_unit, x_train, var='col')

        x_train_unit, voxel_index = select_top(x_train, np.abs(corr), n_voxel, axis=1, verbose=False)
        x_test_unit = x_test[:, voxel_index]

        # Add bias terms
        x_train_unit = add_bias(x_train_unit, axis=1)
        x_test_unit = add_bias(x_test_unit, axis=1)

        # Setup regression
        # For quick demo, use linaer regression
        model = LinearRegression()
        #model = SparseLinearRegression(n_iter=n_iter, prune_mode=1)

        # Training and test
        try:
            model.fit(x_train_unit, y_train_unit)  # Training
            y_pred = model.predict(x_test_unit)    # Test
        except:
            # When SLiR failed, returns zero-filled array as predicted features
            y_pred = np.zeros(y_test_unit.shape)

        # Denormalize predicted features
        y_pred = y_pred * norm_scale_y + norm_mean_y

        y_true_list.append(y_test_unit)
        y_pred_list.append(y_pred)

        print('Time: %.3f sec' % (time() - start_time))

    # Create numpy arrays for return values
    y_predicted = np.vstack(y_pred_list).T
    y_true = np.vstack(y_true_list).T

    return y_predicted, y_true


def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


def cos_sim(v1, v2):
    return np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-8)

def calc_similarity(x, y):
    batch_size = len(x)
    gt_size = len(y)

    similarity = np.empty([batch_size, gt_size])
    for i in range(batch_size):
        for j in range(gt_size):
            similarity[i, j] = cos_sim(x[i], y[j])
    return similarity

def evaluate(Z, Y):
    # Z: (batch_size, 512)
    # Y: (gt_size, 512)
    binary_confusion_matrix = np.zeros([len(Z), len(Y)])
    similarity = calc_similarity(Z, Y)
    acc_tmp = np.zeros(len(similarity))
    for i in range(len(similarity)):
        acc_tmp[i] = np.sum(similarity[i,:] < similarity[i,i]) / (len(similarity)-1)
        binary_confusion_matrix[i,similarity[i,:] < similarity[i,i]] = 1
        binary_confusion_matrix[i,similarity[i,:] > similarity[i,i]] = -1
    similarity_acc = np.mean(acc_tmp)
    # import pdb; pdb.set_trace()
    top10_acc = []
    for i in range(len(similarity)):
        if np.sum(similarity[i,:] > similarity[i,i]) < 10:
            top10_acc.append(1)
        else:
            top10_acc.append(0)
    print('top10 acc: /{}'.format(Y.shape[0]), np.mean(top10_acc))


    print('Similarity Acc', similarity_acc)

    return similarity_acc, binary_confusion_matrix

# Run as a scirpt ######################################################

if __name__ == '__main__':
    # To avoid any use of global variables,
    # do nothing except calling main() here
    main()