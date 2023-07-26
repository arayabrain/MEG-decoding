import os
from typing import List, Tuple
from torch.utils.data import Sampler, Dataset
import numpy as np
import mne
from tqdm import tqdm
from meg_decoding.matlab_utils.load_meg import get_meg_data, roi, time_window, get_baseline
import tqdm
import torch
import h5py
from typing import List
from meg_decoding.utils.preproc_utils import (
    check_preprocs,
    scaleAndClamp,
    scaleAndClamp_single,
    baseline_correction_single,
)
from meg_decoding.video_utils.video_controller import VideoController
from meg_decoding.dataclass.god import get_kernel_block_ids, get_common_kernel
import scipy.io
mne.set_log_level(verbose="WARNING")
import bdpy
from omegaconf import OmegaConf
import pandas as pd
import random
import cv2

def z_score_epoch(epoch:np.ndarray, baseline_mean:np.ndarray=None, baseline_std:np.ndarray=None)->np.ndarray:
    """
    epoch: ch x time
    """
    if baseline_mean is None:
        baseline_mean = np.mean(epoch, axis=1) # 時間方向の平均
    if baseline_std is None:
        baseline_std = np.std(epoch, axis=1) # 時間方向の分散
    return (epoch - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]

def car_epoch(epoch:np.ndarray)->np.ndarray:
    """
    epoch: ch x time
    """
    return epoch - np.mean(epoch, axis=0)

def clamp_epoch(epoch, a_min, a_max):
    """
    epoch: ch x time
    """
    return np.clip(epoch, a_min, a_max)

class SessionDatasetDrama(Dataset):
    force_create_h5:bool = False
    val_length: float = 60 # [s]
    def __init__(self, dataset_config:OmegaConf, preproc_config:OmegaConf, meg_path:str, movie_path:str,
                 movie_trigger_path:str, meg_trigger_path:str, h5_file_name:str, sbj_name:str=None,
                 split:str='train', section_length:float=1, num_section_limit:int=15, image_preprocs:str=[], meg_preprocs:list=[],
                 only_meg:bool=False, on_memory:bool=False, movie_trigger_label:int=3):
        """
        split(ser): all, train, val
        section_length(float): duration of one section [s]
        num_section_limit(int): maximum number of sections for training. (taining only). in case of test, this parameter is ignored and use the last one section.
        h5_file_name(str): h5 file name. it contains preprocessed MEG data.

        dataset_config: has attribute {meg_fs, movie_fs, movie_crop_pt1, movie_crop_pt2, kernel_root, roi_block_ids, ch_region_path}
        preproc_config: has attribute {meg_onset, meg_duration, clamp, bandpass_filter, brain_resample_rate, src_reconstruction, baseline_duration}
        """
        self.dataset_config = dataset_config
        self.preproc_config = preproc_config
        self.meg_fs:float = dataset_config.meg_fs # [Hz]
        self.movie_fs:float = dataset_config.movie_fs # [Hz]
        self.meg_path = meg_path
        self.movie_path = movie_path
        self.movie_trigger_path = movie_trigger_path
        self.meg_trigger_path = meg_trigger_path
        self.split = split
        self.image_preprocs = image_preprocs # list
        self.meg_preprocs = meg_preprocs # list
        self.only_meg = only_meg # bool
        self.on_memory = on_memory # bool
        self.movie_trigger_label = movie_trigger_label
        self.meg_onset:float = preproc_config.meg_onset # [s]
        self.meg_durarion:float = preproc_config.meg_duration # [s]
        self.section_length = section_length # [s]
        self.num_section_limit = num_section_limit # int
        self.h5_file_name = h5_file_name
        self.sbj_name = sbj_name
        self.movie_crop_pt1:Tuple = dataset_config.movie_crop_pt1 # height x width <<--caution
        self.movie_crop_pt2:Tuple = dataset_config.movie_crop_pt2 # height x width <<--caution
        self.clamp:Tuple = self.preproc_config.clamp # a_min, a_max
        assert self.section_length < self.val_length, 'section_length {} must be smaller than val_length {}'.format(self.section_length, self.val_length)
        if self.preproc_config.brain_resample_rate is not None:
            self.decimation_rate:float = self.meg_fs / self.preproc_config.brain_resample_rate
        else:
            self.decimation_rate:float = 1.0
        assert self.decimation_rate >= 1.0, 'decimation_rate {} must be larger than 1.0'.format(self.decimation_rate)

        self.baseline_duration:float = self.preproc_config.baseline_duration # [s]
        self.baseline_duration_frames = int(self.baseline_duration * self.meg_fs / self.decimation_rate) # [frame]
        self.meg_durarion_frames = int(self.meg_durarion * self.meg_fs / self.decimation_rate) # [frame]
        self.meg_onset_frames = int(self.meg_onset * self.meg_fs / self.decimation_rate) # [frame]

    def __len__(self)->int:
        return len(self.indices)

    def __getitem__(self, idx:int)->Tuple:
        target_idx = self.indices[idx] + random.randint(0, self.n_triggers_per_section - 1)
        movie_frame = self.movie_triggers[target_idx]
        meg_frame = self.meg_triggers[target_idx]
        baseline_frame = meg_frame - self.baseline_duration_frames
        end_frame = meg_frame + self.meg_onset_frames + self.meg_durarion_frames

        if self.on_memory:
            ROI_MEG_Data = self.ROI_MEG_Data[:, baseline_frame:end_frame] # ch x time
        else:
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][:,baseline_frame:end_frame]
         # z-score -> baseline correction -> clamp
        ROI_MEG_Data = z_score_epoch(ROI_MEG_Data) # z-score by channel across time
        ROI_MEG_Data -= np.mean(ROI_MEG_Data[:, :self.baseline_duration_frames], axis=1)[:, np.newaxis] # baseline correction
        ROI_MEG_Data = ROI_MEG_Data[:, self.meg_onset_frames:] # remove before onset
        if self.clamp is not None:
            ROI_MEG_Data = clamp_epoch(ROI_MEG_Data, *self.clamp)

        for func_ in self.meg_preprocs:
            ROI_MEG_Data = func_(ROI_MEG_Data)

        if self.only_meg:
            return ROI_MEG_Data # , movie_frame # movie_frame is dummy
        else:
            image = self.vc.get_frame(movie_frame)
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image[self.movie_crop_pt1[0]:self.movie_crop_pt2[0], self.movie_crop_pt1[1]:self.movie_crop_pt2[1], :] # crop
            for func_ in self.image_preprocs:
                image = func_(image)
                # default: image channel is placed at last dimension
            return ROI_MEG_Data, image

    def split_data(self):
        """
        decompose the data into sections.

        if all, this dataset consits of all data.
        elif train, this dataset consists of all data but last 60 sec.
        elif val, this dataset consists of the last 60 sec data.
        """
        self.n_triggers_per_section = int(self.dataset_config.movie_fs / 2 * self.section_length) # [frame] 2 frames for 1 trigger

        movie_triggers = self.get_movie_trigger(self.movie_trigger_path, self.movie_trigger_label)
        meg_triggers = self.get_meg_trigger(self.meg_trigger_path, self.movie_trigger_label, self.decimation_rate)
        assert len(movie_triggers) == len(meg_triggers), 'len(movie_triggers) = {}, len(meg_triggers) = {}'.format(len(movie_triggers), len(meg_triggers))
        val_frames = int(self.dataset_config.movie_fs / 2 * self.val_length) # [frame] 2 frames for 1 trigger
        # first index of each section of val
        self.val_indices = np.arange(len(movie_triggers) - val_frames, len(movie_triggers), self.n_triggers_per_section) # [trigger id]
        self.train_indices = np.arange(0, len(movie_triggers) - val_frames, self.n_triggers_per_section) # [trigger id]
        self.all_indices = np.arange(0, len(movie_triggers), self.n_triggers_per_section) # [trigger id]
        if self.split == 'train':
            self.indices = self.train_indices
        elif self.split == 'val':
            self.indices = self.val_indices
        elif self.split == 'all':
            self.indices = self.all_indices
        else:
            raise ValueError('invalid split {}'.format(self.split))
        if self.num_section_limit is not None:
            self.indices = self.indices[:self.num_section_limit] # necesary random ?
        self.movie_triggers = movie_triggers
        self.meg_triggers = meg_triggers

    def prepare_data(self):
        # 前処理内容
        # CAR -> (src_reconst) -> bandpass filter -> resample
        # 前処理してh5ファイルにする
        MEG_Data:np.ndarray = self.get_meg_matlab_data(self.meg_path)
        MEG_Data = car_epoch(MEG_Data) # common average reference by time
        if not self.only_meg:
            self.vc:VideoController = self.get_video_controler(self.movie_path)
        roi_channels:np.ndarray = roi(self.dataset_config) # electrode id
        if os.path.exists(self.h5_file_name) and not self.force_create_h5:
            print('load h5 file {}'.format(self.h5_file_name))
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][:,:] # ndim =2
        else:
            # -----MEG-----
            ROI_MEG_Data = MEG_Data[roi_channels, :]
            if "src_reconstruction" in self.preproc_config.keys() and self.preproc_config.src_reconstruction:
                assert len(ROI_MEG_Data) == 160, 'get {}'.format(len(ROI_MEG_Data))
                print('src reconstruction')
                target_roi_indices = get_kernel_block_ids(self.dataset_config) # voxel id
                common_kernel_path = os.path.join(self.dataset_config.kernel_root, 'kernel/{sbj}/tess_cortex_pial_low.mat'.format(self.sbj_name))
                subject_kernel_path = os.path.join(self.dataset_config.kernel_root, 'kernel/{sbj}/results_MN_MEG_KERNEL_{}'.format(self.sbj_name, self.meg_path.split('/')[-1]))
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
            if self.preproc_config.brain_resample_rate is not None:
                ROI_MEG_Data = mne.filter.resample(ROI_MEG_Data, down=self.meg_fs / self.preproc_config.brain_resample_rate)
                print('resample {} to {} Hz'.format(self.meg_fs, self.preproc_config.brain_resample_rate))

        assert ROI_MEG_Data.shape[0] == len(roi_channels), 'ROI_MEG_Data.shape[0] = {}, len(roi_channels) = {}'.format(ROI_MEG_Data.shape[0], len(roi_channels))
        assert ROI_MEG_Data.ndim == 2, 'ROI_MEG_Data.ndim = {}'.format(ROI_MEG_Data.ndim)

        if self.on_memory:
            self.ROI_MEG_Data = ROI_MEG_Data
        else:
            with h5py.File(self.h5_file_name, "w") as h5:
                h5.create_dataset("ROI_MEG_Data", data=ROI_MEG_Data)
                print('save ROI_MEG_Data to {}'.format(self.h5_file_name))



    @staticmethod
    def get_meg_matlab_data(meg_path:str)->np.ndarray:
        data = scipy.io.loadmat(meg_path)
        MEG_Data = data['F']
        return MEG_Data

    @staticmethod
    def get_video_controler(movie_path)->VideoController:
        return VideoController(movie_path)

    @staticmethod
    def get_movie_trigger(movie_trigger_path:str, trigger_label:int)->np.ndarray:
        df = pd.read_csv(movie_trigger_path, names=['frame', 'label'])
        trigger = df.query('label=={}'.format(trigger_label))['frame'].to_numpy()
        return trigger

    @staticmethod
    def get_meg_trigger(meg_trigger_path:str, trigger_label:int, decimation_rate:float)->np.ndarray:
        df = pd.read_csv(meg_trigger_path, names=['frame', 'label', 'delay'])
        trigger = df.query('label=={}'.format(trigger_label))['frame'].to_numpy()
        delay = df.query('label=={}'.format(trigger_label))['delay'].to_numpy()
        # 実際の提示タイミングはディジタルトリガよりdelay分だけ遅延している。
        trigger += delay
        trigger /= decimation_rate
        trigger = np.floor(trigger).astype(np.int)
        return trigger
