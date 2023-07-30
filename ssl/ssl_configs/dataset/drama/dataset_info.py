import os
import numpy as np


MEGROOT = '/work/project/MEG_GOD/GOD_dataset/VideoWatching'
processed_meg_path_pattern = os.path.join(MEGROOT, 'preprocess_MEG/{sub}/{session_name}')
meg_trigger_path_pattern = '/home/yainoue/meg2image/codes/matlab_codes/triggers/{sub}/onset_trigger_video_{video_id}-part_{part_id}.csv'

VIDEO_ROOT  = '/storage/dataset/MEG/internal/AnnotatedMovie_v1/tmp/stim_video/'
movie_file_list = [
    'ID01_HerosVol1-1_id1_MEG_DATAPixx_part{part_id}.mp4',
    'ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part{part_id}.mp4',
    'ID03_GleeVol1-1_id3_MEG_DATAPixx_part{part_id}.mp4',
    'ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part{part_id}.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part{part_id}.mp4',
    'ID06_TheBigBangTheoryVol1-1_id6_MEG_DATAPixx_part{part_id}.mp4',
    'ID07_TheBigBangTheoryVol1-2_id7_MEG_DATAPixx_part{part_id}.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part{part_id}.mp4',
    'ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part{part_id}.mp4',
    'ID10_GhostInTheShellVol1-1_id10_MEG_DATAPixx_part{part_id}.mp4'

]
movie_trigger_path_pattern = '/home/yainoue/meg2image/codes/MEG-decoding/data/movies/video{video_id}-part{part_id}_onset_triggers.csv'

num_parts = [4, 3, 4, 4, 6, 2, 2, 9, 4, 2]
assert len(num_parts)==10, 'length of num_parts must be 10'
assert np.sum(num_parts) == 40

session2video_id = []
session2part_id = []
for i, num_part in enumerate(num_parts):
    video_id = i+1
    session2video_id += [video_id] * num_part
    session2part_id += list(range(1, num_part+1))

assert len(session2part_id) == np.sum(num_parts) , 'length of session2part_id must be {}'.format(np.sum(num_parts))
assert len(session2video_id) == np.sum(num_parts) , 'length of session2video_id must be {}'.format(np.sum(num_parts))


def get_dataset_info(name:str, h5_dir:str):
    """_summary_

    Args:
        name (str): sbj_x_y_z-session_{id1}_{id2}...
        split (str): train or val
    """
    sbjs, session_ids = name.split('-')
    sbjs = sbjs.split('_')[1:]
    assert len(sbjs) > 0, 'sbjs must be list'
    session_ids = session_ids.replace('session_', '')
    if '~' in session_ids:
        start_id, end_id = session_ids.split('~')
        session_ids = list(range(start_id, end_id+1))
    elif session_ids == 'all':
        session_ids = list(range(40))
    else:
        session_ids = session_ids.split('_')
    assert len(session_ids) > 0, 'session_ids must be list or range'
    dataset_info_list = []
    for sbj in sbjs:
        for session_id in session_ids:
            video_id = session2video_id[session_id]
            part_id = session2part_id[session_id]
            ret = dataset_path(sbj, video_id, part_id, h5_dir)
            dataset_info_list.append(ret)
    print('=================GOD=================')
    print(name)
    print('dataset_info_list: ', dataset_info_list)
    print('=====================================')
    return dataset_info_list

def dataset_path(sbj:int, video_id:int, part_id:int, h5_dir):
    sbj = 'sbj{}'.format(str(sbj).zfill(2))
    movie_name = movie_file_list[video_id-1].format(part_id=part_id)

    ret = {
        'meg_path': processed_meg_path_pattern.format(sub=sbj, session_name=movie_name.replace('.mp4', '.mat')),
        'movie_path': os.path.join(VIDEO_ROOT, movie_name),
        'movie_trigger_path': movie_trigger_path_pattern.format(sub=sbj, video_id=video_id, part_id=part_id),
        'meg_trigger_path': meg_trigger_path_pattern.format(sub=sbj, video_id=video_id, part_id=part_id),
        'sbj_name': sbj,
        'h5_file_name': os.path.join(h5_dir, '{}_{}.h5'.format(sbj, movie_name)),
    }
    return ret
