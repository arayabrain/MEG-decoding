import os


DATAROOT = '/work/project/MEG_GOD/GOD_dataset/'
processed_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{session_name}')
label_path_pattern = os.path.join(DATAROOT, '{sub}/labels/{session_name}')
trigger_meg_path_pattern = os.path.join(DATAROOT, '{sub}/trigger/{session_name}')
processed_rest_meg_path_pattern = os.path.join(DATAROOT, '{sub}/mat/{session_name}')
image_root = '/storage/dataset/ECoG/internal/GODv2-4/'
image_id_path_pattern = os.path.join(DATAROOT, 'clip_image_{split_relate}.mat')

def get_dataset_info(name:str, h5_dir:str):
    """_summary_

    Args:
        name (str): sbj_x_y_z-{split}-session_{id1}_{id2}...
        split (str): train or val
    """
    sbjs, split, session_ids = name.split('-')
    sbjs = sbjs.split('_')[1:]
    assert len(sbjs) > 0, 'sbjs must be list'
    session_ids = session_ids.replace('session_', '')
    if '~' in session_ids:
        start_id, end_id = session_ids.split('~')
        session_ids = list(range(start_id, end_id+1))
    elif session_ids == 'all':
        session_ids = list(range(1, 12+1)) # 1-12までがGODのsession
    else:
        session_ids = session_ids.split('_')
    assert len(session_ids) > 0, 'session_ids must be list or range'
    dataset_info_list = []
    for sbj in sbjs:
        for session_id in session_ids:
            ret = dataset_path(sbj, split, session_id, h5_dir)
            dataset_info_list.append(ret)
    print('=================GOD=================')
    print(name)
    print('dataset_info_list: ', dataset_info_list)
    print('=====================================')
    return dataset_info_list

def dataset_path(sbj, split, id, h5_dir):
    sbj = 'sbj{}'.format(str(sbj).zfill(2))
    if split == 'train':
        image_dir = os.path.join(image_root, 'images_trn')
        session_name = 'data_block{}'.format(str(id).zfill(3))
        split_relate = 'training'
    elif split == 'val':
        image_dir = os.path.join(image_root, 'images_val')
        session_name = 'data_val{}'.format(str(id).zfill(3))
        split_relate = 'test'
    else:
        raise ValueError('split must be train or val')

    ret = {
        'image_root': image_dir,
        'meg_path': trigger_meg_path_pattern.format(sub=sbj, session_name=session_name),
        'meg_label_path': label_path_pattern.format(sub=sbj, session_name=session_name),
        'meg_trigger_path': trigger_meg_path_pattern.format(sub=sbj, session_name=session_name),
        'meg_rest_path': processed_rest_meg_path_pattern.format(sub=sbj, session_name=session_name),
        'sbj_name': sbj,
        'h5_file_name': os.path.join(h5_dir, '{}_{}.h5'.format(sbj, session_name)),
        'image_id_path': image_id_path_pattern.format(split_relate=split_relate),
    }
    return ret
