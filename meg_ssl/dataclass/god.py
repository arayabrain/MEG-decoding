import os
from typing import List, Tuple
from torch.utils.data import Sampler, Dataset
import numpy as np
import mne
from tqdm import tqdm
from meg_decoding.matlab_utils.load_meg import roi
import h5py
from meg_decoding.video_utils.video_controller import VideoController
from meg_decoding.dataclass.god import get_kernel_block_ids, get_common_kernel
from .utils import z_score_epoch, car_epoch, clamp_epoch
import scipy.io
mne.set_log_level(verbose="WARNING")
import bdpy
from omegaconf import OmegaConf
import pandas as pd
import random
import cv2
import mat73

class SessionDatasetGOD(Dataset):
    force_create_h5:bool = False

    def __init__(self, dataset_config:OmegaConf, preproc_config:OmegaConf, meg_path:str, image_root:str,
                 meg_trigger_path:str, meg_label_path:str, h5_file_name:str, image_id_path:str, sbj_name:str=None,
                 image_preprocs:str=[], meg_preprocs:list=[], num_trial_limit:int=1200,
                 only_meg:bool=False, on_memory:bool=False, ret_image_label:bool=False):
        """
        dataset_config: has attribute {meg_fs, kernel_root, roi_block_ids, ch_region_path, region}
        preproc_config: has attribute {meg_onset, meg_duration, clamp, bandpass_filter, brain_resample_rate, src_reconstruction, baseline_duration}
        """
        self.dataset_config = dataset_config
        self.preproc_config = preproc_config
        self.meg_fs:float = dataset_config.meg_fs # [Hz]
        self.meg_path = meg_path
        self.image_root = image_root
        self.meg_label_path = meg_label_path
        self.image_id_path = image_id_path
        self.meg_trigger_path = meg_trigger_path
        self.image_preprocs = image_preprocs # list
        self.meg_preprocs = meg_preprocs # list
        self.only_meg = only_meg # bool
        self.on_memory = on_memory # bool
        self.ret_image_label = ret_image_label # bool
        self.meg_onset:float = preproc_config.meg_onset # [s]
        self.meg_durarion:float = preproc_config.meg_duration # [s]
        self.h5_file_name = h5_file_name
        self.sbj_name = sbj_name
        self.num_image_limit =num_trial_limit
        self.clamp:Tuple = self.preproc_config.clamp # a_min, a_max
        if self.preproc_config.brain_resample_rate is not None:
            self.decimation_rate:float = self.meg_fs / self.preproc_config.brain_resample_rate
        else:
            self.decimation_rate:float = 1.0
        assert self.decimation_rate >= 1.0, 'decimation_rate {} must be larger than 1.0'.format(self.decimation_rate)

        self.baseline_duration:float = self.preproc_config.baseline_duration # [s]
        self.baseline_duration_frames = int(self.baseline_duration * self.meg_fs / self.decimation_rate) # [frame]
        self.meg_duration_frames = int(np.round(self.meg_durarion * self.meg_fs / self.decimation_rate)) # [frame]
        self.meg_onset_frames = int(self.meg_onset * self.meg_fs / self.decimation_rate) # [frame]

        self.prepare_data()


    def __len__(self)->int:
        return len(self.indices)

    def __getitem__(self, idx:int)->Tuple:
        meg_frame = self.meg_triggers[idx]
        baseline_frame = meg_frame - self.baseline_duration_frames
        end_frame = meg_frame + self.meg_onset_frames + self.meg_duration_frames
        if self.on_memory:
            ROI_MEG_Data = self.ROI_MEG_Data[:, baseline_frame:end_frame] # ch x time
        else:
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][:,baseline_frame:end_frame]
         # z-score -> baseline correction -> clamp
        ROI_MEG_Data = z_score_epoch(ROI_MEG_Data) # z-score by channel across time
        ROI_MEG_Data -= np.mean(ROI_MEG_Data[:, :self.baseline_duration_frames], axis=1)[:, np.newaxis] # baseline correction
        ROI_MEG_Data = ROI_MEG_Data[:, -self.meg_duration_frames:]# ROI_MEG_Data[:, self.meg_onset_frames:] # remove before onset
        if self.clamp is not None:
            ROI_MEG_Data = clamp_epoch(ROI_MEG_Data, *self.clamp)

        for func_ in self.meg_preprocs:
            ROI_MEG_Data = func_(ROI_MEG_Data)

        if self.only_meg:
            return ROI_MEG_Data # , movie_frame # movie_frame is dummy
        else:
            image_idx = self.meg_labels[idx]-1  # idxはあくまでもセッションにおける順番
            image_path = os.path.join(self.image_root, self.image_names[image_idx])
            image = cv2.imread(image_path)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for func_ in self.image_preprocs:
                image = func_(image)
                # default: image channel is placed at last dimension
            if self.ret_image_label:
                return ROI_MEG_Data, image, image_idx
            else:
                return ROI_MEG_Data, image



    def prepare_data(self):
        # 前処理内容
        # CAR -> (src_reconst) -> bandpass filter -> resample
        # 前処理してh5ファイルにする
        MEG_Data:np.ndarray = self.get_meg_matlab_data(self.meg_path)
        # MEG_Data = car_epoch(MEG_Data) # common average reference by time
        if not self.only_meg:
            self.image_names = self.get_image_name_list(self.image_id_path)
        roi_channels:np.ndarray = roi(self.dataset_config) # electrode id
        if os.path.exists(self.h5_file_name) and not self.force_create_h5:
            print('load h5 file {}'.format(self.h5_file_name))
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][:,:] # ndim =2
        else:
            # -----MEG-----
            ROI_MEG_Data = MEG_Data[roi_channels, :]
            ROI_MEG_Data = car_epoch(ROI_MEG_Data) # common average reference by time
            if self.preproc_config.src_reconstruction:
                assert len(ROI_MEG_Data) == 160, 'get {}'.format(len(ROI_MEG_Data))
                print('src reconstruction')
                target_roi_indices = get_kernel_block_ids(self.dataset_config) # voxel id
                common_kernel_path = os.path.join(self.dataset_config.kernel_root, '{sbj}/kernel/tess_cortex_pial_low.mat'.format(self.sbj_name))
                subject_kernel_path = os.path.join(self.dataset_config.kernel_root, '{sbj}/kernel/results_MN_MEG_KERNEL_{}'.format(self.sbj_name, self.meg_path.split('/')[-1]))
                target_region_kernel = get_common_kernel(target_roi_indices, subject_kernel_path, common_kernel_path)

                ROI_MEG_Data = np.matmul(target_region_kernel, ROI_MEG_Data) # source reconstruction of target region
                print('apply kernel for source reconstruction')
            else:
                pass
            if self.preproc_config.bandpass_filter is not None:
                bandpass_filter_low = self.preproc_config.bandpass_filter[0]
                bandpass_filter_high = self.preproc_config.bandpass_filter[1]
                ROI_MEG_Data = mne.filter.filter_data(ROI_MEG_Data, sfreq=self.meg_fs, l_freq=bandpass_filter_low, h_freq=bandpass_filter_high,)
                print(f'band path filter: {bandpass_filter_low}-{bandpass_filter_high}')
            if (self.preproc_config.brain_resample_rate is not None) or (self.preproc_config.brain_resample_rate<self.meg_fs):
                ROI_MEG_Data = mne.filter.resample(ROI_MEG_Data, down=self.meg_fs / self.preproc_config.brain_resample_rate)
                print('resample {} to {} Hz'.format(self.meg_fs, self.preproc_config.brain_resample_rate))

        assert ROI_MEG_Data.shape[0] == len(roi_channels), 'ROI_MEG_Data.shape[0] = {}, len(roi_channels) = {}'.format(ROI_MEG_Data.shape[0], len(roi_channels))
        assert ROI_MEG_Data.ndim == 2, 'ROI_MEG_Data.ndim = {}'.format(ROI_MEG_Data.ndim)

        ROI_MEG_Data = ROI_MEG_Data.astype(np.float32)
        if self.on_memory:
            self.ROI_MEG_Data = ROI_MEG_Data
        else:
            if os.path.exists(self.h5_file_name) and not self.force_create_h5:
                pass
            else:
                with h5py.File(self.h5_file_name, "w") as h5:
                    h5.create_dataset("ROI_MEG_Data", data=ROI_MEG_Data)
                    print('save ROI_MEG_Data to {}'.format(self.h5_file_name))

        self.meg_triggers = self.get_meg_trigger(self.meg_trigger_path, self.decimation_rate, self.meg_fs) # [frame]
        self.meg_labels = self.get_meg_label(self.meg_label_path) # [frame]
        self.indices = np.arange(len(self.meg_triggers))[:self.num_image_limit]

        self.num_electrodes = ROI_MEG_Data.shape[0]



    @staticmethod
    def get_image_name_list(image_id_path)->List:
        data = mat73.loadmat(image_id_path)
        return data['label']

    @staticmethod
    def get_meg_label(label_path:str)->np.ndarray:
        data = scipy.io.loadmat(label_path)
        return data['vec_index'][0] # 1始まり

    @staticmethod
    def get_meg_trigger(meg_trigger_path:str, decimation_rate:float, fs:float)->np.ndarray:
        trigger = scipy.io.loadmat(meg_trigger_path) # [sec]
        trigger = trigger['trigger'][0] # [sec]
        trigger = trigger * fs
        trigger /= decimation_rate
        trigger = np.floor(trigger).astype(np.int)
        return trigger

    @staticmethod
    def get_meg_matlab_data(meg_path)->np.ndarray:
        data = scipy.io.loadmat(meg_path) # ch x time sample # 203 x 391000
        MEG_Data = data['F']
        assert len(MEG_Data) == 203
        return MEG_Data