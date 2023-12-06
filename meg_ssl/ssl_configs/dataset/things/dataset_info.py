import os


DATAROOT = '/work/project/MEG_GOD/yainoue/things/'
meg_root = os.path.join(DATAROOT, 'MEG')
image_root = os.path.join(DATAROOT, 'Images')

def get_dataset_info(name:str, h5_dir:str)->list:
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
        session_ids = list(range(int(start_id), int(end_id+1)))
    elif session_ids == 'all':
        if split == 'train':
            session_ids = list(range(1, 12+1)) # 1-12までがGODのtest session
        elif split == 'val':
            session_ids = list(range(1, 12+1)) # 1-12までがGODのtest session
        else:
            raise ValueError('split only takes train or val')
    else:
        session_ids = session_ids.split('_')
    assert len(session_ids) > 0, 'session_ids must be list or range'
    dataset_info_list = []
    for sbj in sbjs:
        for session_id in session_ids:
            ret = dataset_path(sbj, split, session_id, h5_dir)
            dataset_info_list.append(ret)
    print('=================THINGS=================')
    print(name)
    print('dataset_info_list: ', dataset_info_list)
    print('=====================================')
    return dataset_info_list

def dataset_path(sbj, split, id, h5_dir)->dict:
    # sbj = 'sbj{}'.format(str(sbj).zfill(2))
    if split == 'train':
        session_name = 'data_block{}'.format(str(id).zfill(3))
        split_name = 'exp'
    elif split == 'val':
        session_name = 'data_val{}'.format(str(id).zfill(3))
        split_name = 'test'
    else:
        raise ValueError('split must be train or val')

    ret = {
        'session_id': id,
        'image_root': image_root,
        'meg_root': meg_root,
        'h5_file_name': os.path.join(h5_dir, '{}_{}_{}.h5'.format(sbj, session_name, split_name)),
        'sbj_name': sbj,
        'split':split_name
    }
    return ret
