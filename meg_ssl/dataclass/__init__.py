import os
from omegaconf import OmegaConf
from hydra import compose, initialize
from typing import List, Tuple, Dict, Union
from torch.utils.data import Sampler, Dataset, ConcatDataset
from .drama import SessionDatasetDrama
from .god import SessionDatasetGOD
from torch.utils.data import ConcatDataset
from meg_ssl.ssl_configs.dataset.drama.dataset_info import get_dataset_info as get_drama_dataset_info
from meg_ssl.ssl_configs.dataset.GOD.dataset_info import get_dataset_info as get_god_dataset_info


BLACK_MAT_FILE_LIST = ['sbj01/ID08_DreamGirlsVol1_1_id8_MEG_DATAPixx_part4.mat',
                       'sbj03/ID08_DreamGirlsVol1_1_id8_MEG_DATAPixx_part5.mat']

def parse_dataset(dataset_names:dict, dataset_yamls:dict, preproc_config:OmegaConf,
                  num_trial_limits_dict:dict, h5_root:str, image_preprocs:list=[], meg_preprocs:list=[],
                  only_meg:bool=False, on_memory:bool=False, ret_image_label:bool=False)->dict:
    """_summary_

    Args:
        dataset_names (dict): {'train':{'drama': sbj_1_session_all}, 'val': {'god: sbj_1_session_6_12}}}
        dataset_yamls (dict): {'drama': drama_vc.yaml, 'god': god_vc.yaml}
        num_trial_limits (dict): {'train':{'drama':12000}, 'val':{'god': 1200}}
    """
    split_datasets = {}
    with initialize(config_path="../ssl_configs/dataset"):
        for split, name_dict in dataset_names.items():
            dataset_info_list = []
            dataset_config_list = []
            num_trial_limits = []
            for name, session_info in name_dict.items():
                h5_dir = os.path.join(h5_root, name)
                os.makedirs(h5_dir, exist_ok=True)
                try:
                    cfg = compose(config_name=dataset_yamls[name])[name]
                except KeyError:
                    print('dataset config should locate under its name dir')
                    raise Exception()
                if name == 'drama':
                    tmp_dataset_info_list = get_drama_dataset_info(session_info, h5_dir, split)
                elif name == 'GOD':
                    tmp_dataset_info_list = get_god_dataset_info(session_info, h5_dir)
                else:
                    raise ValueError('name {} is not supported'.format(name))
                

                dataset_info_list += tmp_dataset_info_list
                dataset_config_list += [cfg] * len(dataset_info_list)
                num_trial_limits += [int(num_trial_limits_dict[split][name]/len(dataset_info_list))] * len(dataset_info_list)

            split_datasets[split] = collect_session_dataset(dataset_info_list, dataset_config_list, preproc_config,
                        num_trial_limits, image_preprocs, meg_preprocs, only_meg, on_memory, ret_image_label)
    return split_datasets


def collect_session_dataset(dataset_info_list:List[Dict], dataset_config_list:List[OmegaConf], preproc_config:OmegaConf,
                    num_trial_limits:list, image_preprocs:list=[], meg_preprocs:list=[], only_meg:bool=False, on_memory:bool=False, ret_image_label:bool=False):
    dataset_list = []
    for dataset_info, dataset_config, num_trial_limit in zip(dataset_info_list, dataset_config_list, num_trial_limits):
        dataset = get_session_dataset(dataset_info, dataset_config, preproc_config, num_trial_limit, image_preprocs, meg_preprocs, only_meg, on_memory, ret_image_label)
        skip_flag = 0
        for black in BLACK_MAT_FILE_LIST:
            if black in dataset.meg_path:
                skip_flag = 1
                break
        if skip_flag == 1:
            print('skipping the following dataset: ', dataset.meg_path)
            continue
        dataset_list.append(dataset)

    return ConcatDataset(dataset_list)



def get_session_dataset(dataset_info:dict, dataset_config:OmegaConf, preproc_config:OmegaConf,
                        num_trial_limit, image_preprocs, meg_preprocs,
                        only_meg, on_memory, ret_image_label)->Union[SessionDatasetDrama, SessionDatasetGOD]:
    if dataset_config.name == 'drama':
        return SessionDatasetDrama(dataset_config, preproc_config, dataset_info['meg_path'], dataset_info['movie_path'],
                                   dataset_info['movie_trigger_path'], dataset_info['meg_trigger_path'], dataset_info['h5_file_name'],
                                   dataset_info['movie_crop_pts'], sbj_name=dataset_info['sbj_name'], split=dataset_info['split'], 
                                   num_trial_limit=num_trial_limit, image_preprocs=image_preprocs, meg_preprocs=meg_preprocs,
                                   only_meg=only_meg, on_memory=on_memory)
    elif dataset_config.name == 'GOD':
        return SessionDatasetGOD(dataset_config, preproc_config, dataset_info['meg_path'], dataset_info['image_root'],
                                 dataset_info['meg_trigger_path'], dataset_info['meg_label_path'], dataset_info['h5_file_name'],
                                 dataset_info['image_id_path'], sbj_name=dataset_info['sbj_name'], image_preprocs=image_preprocs, meg_preprocs=meg_preprocs,
                                 num_trial_limit=num_trial_limit, only_meg=only_meg, on_memory=on_memory, ret_image_label=ret_image_label)
    else:
        raise ValueError('dataset_config.name {} is not supported'.format(dataset_config.name))