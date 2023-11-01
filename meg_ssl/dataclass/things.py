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


#*****************************#
### HELPER FUNCTIONS ###  from  https://github.com/ViCCo-Group/THINGS-data/tree/main
#*****************************#
def setup_paths(meg_dir, session):
    run_paths,event_paths = [],[]
    for file in os.listdir(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/'):
        if file.endswith(".ds") and file.startswith("sub"):
            run_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
        if file.endswith("events.tsv") and file.startswith("sub"):
            event_paths.append(os.path.join(f'{meg_dir}/ses-{str(session).zfill(2)}/meg/', file))
    run_paths.sort()
    event_paths.sort()

    return run_paths, event_paths

def read_raw(curr_path,session,run,participant):
    raw = mne.io.read_raw_ctf(curr_path,preload=True)
    # signal dropout in one run -- replacing values with median
    if participant == '1' and session == 11 and run == 4:
        n_samples_exclude   = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-13.4)):np.argmin(np.abs(raw.times-13.4))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T
    elif participant == '2' and session == 10 and run == 2:
        n_samples_exclude = int(0.2/(1/raw.info['sfreq']))
        raw._data[:,np.argmin(np.abs(raw.times-59.8)):np.argmin(np.abs(raw.times-59.8))+n_samples_exclude] = np.repeat(np.median(raw._data,axis=1)[np.newaxis,...], n_samples_exclude, axis=0).T

    raw.drop_channels('MRO11-1609')

    return raw

def read_events(event_paths,run,raw, trigger_channel, trigger_amplitude):
    # load event file that has the corrected onset times (based on optical sensor and replace in the events variable)
    event_file = pd.read_csv(event_paths[run],sep='\t')
    event_file.value.fillna(999999,inplace=True)
    events = mne.find_events(raw, stim_channel=trigger_channel,initial_event=True)
    events = events[events[:,2]==trigger_amplitude]
    events[:,0] = event_file['sample']
    events[:,2] = event_file['value']
    return events

def concat_epochs(raw, events, epochs, pre_stim_time, post_stim_time):
    if epochs:
        epochs_1 = mne.Epochs(raw, events, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
        epochs_1.info['dev_head_t'] = epochs.info['dev_head_t']
        epochs = mne.concatenate_epochs([epochs,epochs_1])
    else:
        epochs = mne.Epochs(raw, events, tmin = pre_stim_time, tmax = post_stim_time, picks = 'mag',baseline=None)
    return epochs

def baseline_correction(epochs):
    baselined_epochs = mne.baseline.rescale(data=epochs.get_data(),times=epochs.times,baseline=(None,0),mode='zscore',copy=False)
    epochs = mne.EpochsArray(baselined_epochs, epochs.info, epochs.events, epochs.tmin,event_id=epochs.event_id)
    return epochs

def stack_sessions(sourcedata_dir,preproc_dir,participant,session_epochs,output_resolution):
    for epochs in session_epochs:
        epochs.info['dev_head_t'] = session_epochs[0].info['dev_head_t']
    all_epochs = mne.concatenate_epochs(epochs_list = session_epochs, add_offset=True)
    all_epochs.metadata = pd.read_csv(f'{sourcedata_dir}/sample_attributes_P{str(participant)}.csv')
    all_epochs.decimate(decim=(1200/output_resolution))
    all_epochs.save(f'{preproc_dir}/preprocessed_P{str(participant)}-epo.fif', overwrite=True)
    print(all_epochs.info)

def save_per_sessions(sourcedata_dir,preproc_dir,participant,session_epochs,output_resolution, sess_list):
    start_id = 0
    for sess, epochs in zip(sess_list, session_epochs):
        end_id = start_id + len(epochs)
        metadata = pd.read_csv(f'{sourcedata_dir}/sample_attributes_P{str(participant)}.csv')
        epochs.metadata = metadata.iloc[start_id:end_id]
        epochs.decimate(decim=(1200/output_resolution))
        savefile = f'{preproc_dir}/preprocessed_P{str(participant)}-epo-sess{sess}.fif'
        epochs.save(savefile, overwrite=True)
        start_id = end_id
    print(savefile)


#*****************************#
### FUNCTION TO RUN PREPROCESSING ###
#*****************************#
def run_preprocessing(meg_dir,session,participant, trigger_channel, trigger_amplitude, l_freq, h_freq, pre_stim_time, post_stim_time):
    epochs = []
    run_paths, event_paths = setup_paths(meg_dir, session)
    for run, curr_path in enumerate(run_paths):
        raw = read_raw(curr_path,session,run, participant)
        events = read_events(event_paths,run,raw, trigger_channel, trigger_amplitude)
        raw.filter(l_freq=l_freq,h_freq=h_freq)
        epochs = concat_epochs(raw, events, epochs, pre_stim_time, post_stim_time)
        epochs.drop_bad()
    print(epochs.info)
    epochs = baseline_correction(epochs)
    return epochs



class SessionDatasetThings(Dataset):
    force_create_h5:bool = False
    trigger_channel:str = 'UPPT001'
    trigger_amplitude           = 64
    num_epochs_per_session = 2254
    original_fs = 1200
    num_test_epochs_per_session = 200
    num_catch_epochs_per_session = 200
    num_exp_epochs_per_session = num_epochs_per_session - num_test_epochs_per_session - num_catch_epochs_per_session
    def __init__(self, dataset_config:OmegaConf, preproc_config:OmegaConf, session_id:str, image_root:str,
                 meg_root:str, h5_file_name:str, sbj_name:int=None,
                 image_preprocs:str=[], meg_preprocs:list=[], num_trial_limit:int=1200,
                 only_meg:bool=False, on_memory:bool=False, ret_image_label:bool=False, split:str='exp'):
        """
        dataset_config: has attribute {meg_fs, kernel_root, roi_block_ids, ch_region_path, region}
        preproc_config: has attribute {meg_onset, meg_duration, clamp, bandpass_filter, brain_resample_rate, src_reconstruction, baseline_duration}
        split: exp, catch, test
        """
        print('session id is ', session_id)
        if split == 'exp':
            print('split is exp')
        elif split == 'test':
            print('split is test')
        elif split == 'catch':
            print('catch: images are not prepared')
            raise ValueError()
        else:
            raise ValueError()
        self.dataset_config = dataset_config
        self.preproc_config = preproc_config
        self.meg_fs:float = dataset_config.meg_fs # [Hz]
        assert self.meg_fs == self.original_fs, 'meg_fs must be 1200'
        self.session_id = session_id
        self.image_root = image_root
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
        self.split:str = split
        if self.preproc_config.brain_resample_rate is not None:
            self.decimation_rate:float = self.meg_fs / self.preproc_config.brain_resample_rate
        else:
            self.decimation_rate:float = 1.0
        assert self.decimation_rate >= 1.0, 'decimation_rate {} must be larger than 1.0'.format(self.decimation_rate)

        self.baseline_duration:float = self.preproc_config.baseline_duration # [s]
        self.baseline_duration_frames = int(self.baseline_duration * self.meg_fs / self.decimation_rate) # [frame]
        # self.meg_duration_frames = int(np.round(self.meg_durarion * self.meg_fs / self.decimation_rate)) # [frame]
        self.meg_onset_frames = int(self.meg_onset * self.meg_fs / self.decimation_rate) # [frame]


        self.meg_dir                     = f'{meg_root}/sub-BIGMEG{sbj_name}/'
        self.sourcedata_dir              = f'{meg_root}/sourcedata/'
        self.preproc_dir                 = f'{meg_root}/derivatives/preprocessed/'

        self.prepare_data()


    def __len__(self)->int:
        return len(self.indices)

    def __getitem__(self, idx:int)->Tuple:
        idx = self.indices[idx]
        # [n_epochs, n_channels, n_times]
        if self.on_memory:
            ROI_MEG_Data = self.ROI_MEG_Data[idx, :, :] # ch x time
        else:
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][idx, :, :]
         # z-score -> baseline correction -> clamp
        ROI_MEG_Data = z_score_epoch(ROI_MEG_Data) # z-score by channel across time
        ROI_MEG_Data -= np.mean(ROI_MEG_Data[:, :self.baseline_duration_frames], axis=1)[:, np.newaxis] # baseline correction
        # ROI_MEG_Data = ROI_MEG_Data[:, -self.meg_duration_frames:]# ROI_MEG_Data[:, self.meg_onset_frames:] # remove before onset
        if self.clamp is not None:
            ROI_MEG_Data = clamp_epoch(ROI_MEG_Data, *self.clamp)

        for func_ in self.meg_preprocs:
            ROI_MEG_Data = func_(ROI_MEG_Data)

        if self.only_meg:
            return ROI_MEG_Data # , movie_frame # movie_frame is dummy
        else:
            image_path = os.path.join(self.image_root, self.image_names[idx])
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cat_id = self.cat_ids[idx]
            for func_ in self.image_preprocs:
                image = func_(image)
                # default: image channel is placed at last dimension
            if self.ret_image_label:
                return ROI_MEG_Data, image, cat_id
            else:
                return ROI_MEG_Data, image



    def prepare_data(self):

        # 前処理内容
        # CAR -> (src_reconst) -> bandpass filter -> resample
        ## Kingの前処理は downsamlple(120Hz) -> clip -> epoching -> baseline correction (-mean signal)

        full_metadata = pd.read_csv(f'{self.sourcedata_dir}/sample_attributes_P{str(self.sbj_name)}.csv')
        assert len(epochs) == self.num_epochs_per_session
        metadata = full_metadata.query(f'session_nr=={self.session_id}')# metadata.iloc[start_id:end_id]
        image_names = []
        cat_names = []
        cat_ids = []
        for image_name in metadata.query(f'trial_type=="{self.split}"')['image_path'].to_list():
            # images_meg/notebook/notebook_03s.jpg',
            # images_catch_meg/catch047.jpg'
            # images_test_meg/tuxedo_16s.jpg

            base_name =  image_name.split('/')[-1]
            cat_name =base_name.split('_')[0]
            image_name = os.path.join(cat_name, base_name)
            image_names.append(image_name)
            cat_names.append(cat_name)

        for cat_id in metadata.query(f'trial_type=="{self.split}"')['things_category_nr'].to_list():
            cat_ids.append(cat_id)

        if os.path.exists(self.h5_file_name) and not self.force_create_h5:
            print('load h5 file {}'.format(self.h5_file_name))
            with h5py.File(self.h5_file_name, "r") as h5:
                ROI_MEG_Data = h5['ROI_MEG_Data'][:,:] # ndim =2
        else:
            # -----MEG-----
            # TODO: able to select ROI
            epochs = []
            run_paths, event_paths = setup_paths(self.meg_dir, self.session_id)
            for run, curr_path in enumerate(run_paths):
                raw = read_raw(curr_path, self.session_id,run, self.sbj_name)
                events = read_events(event_paths,run,raw, self.trigger_channel, self.trigger_amplitude)
                raw.filter(l_freq=self.preproc_config.bandpass_filter[0],h_freq=self.preproc_config.bandpass_filter[1])
                epochs = concat_epochs(raw, events, epochs, self.preproc_config.meg_onset, self.preproc_config.meg_onset+self.preproc_config.meg_duration)
                epochs.drop_bad()
            # print(epochs.info)
            # epochs = baseline_correction(epochs)
            assert len(metadata) == self.num_epochs_per_session
            epochs.metadata = metadata
            epochs.decimate(decim=(self.original_fs/self.preproc_config.brain_resample_rate))


            ignore_indices =[]
            for split_name in ['exp', 'test', 'catch']:
                if split_name == self.split:
                    target_indices = metadata.query(f'trial_type=="{split_name}"').index.to_numpy()
                else:
                    ignore_indices += metadata.query(f'trial_type=="{split_name}"').index.to_list()

            epochs = epochs.drop(ignore_indices)
            self.num_electrodes = epochs.info['nchan']


            if self.split == 'exp':
                n_samples = self.num_exp_epochs_per_session
            elif self.split == 'test':
                n_samples = self.num_test_epochs_per_session
            elif self.split == 'catch':
                n_samples = self.num_catch_epochs_per_session
            assert len(image_names) == len(epochs)
            assert len(image_names) == n_samples

            ROI_MEG_Data = epochs.get_data() # [n_epochs, n_channels, n_times] ex) (2254, 271, 1681)
            assert ROI_MEG_Data.shape[1] == self.num_electrodes
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
        self.num_electrodes = ROI_MEG_Data.shape[1]
        self.image_names = image_names
        self.cat_names = cat_names
        self.cat_ids = cat_ids

        self.indices = np.arange(len(ROI_MEG_Data))[:self.num_image_limit]

