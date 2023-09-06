import cv2
import os
try:
    from meg_decoding.video_utils.video_controller import VideoController
except ModuleNotFoundError:
    import sys
    sys.path.append('.')
    from meg_decoding.video_utils.video_controller import VideoController

save_root = '../../dataset/drama_image'
file_root = '/storage/dataset/MEG/internal/AnnotatedMovie_v1/tmp/stim_video'
filename_list = [
    'ID01_HerosVol1-1_id1_MEG_DATAPixx_part1.mp4',
    'ID01_HerosVol1-1_id1_MEG_DATAPixx_part2.mp4',
    'ID01_HerosVol1-1_id1_MEG_DATAPixx_part3.mp4',
    'ID01_HerosVol1-1_id1_MEG_DATAPixx_part4.mp4',
    'ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part1.mp4',
    'ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part2.mp4',
    'ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part3.mp4',
    'ID03_GleeVol1-1_id3_MEG_DATAPixx_part1.mp4',
    'ID03_GleeVol1-1_id3_MEG_DATAPixx_part2.mp4',
    'ID03_GleeVol1-1_id3_MEG_DATAPixx_part3.mp4',
    'ID03_GleeVol1-1_id3_MEG_DATAPixx_part4.mp4',
    'ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part1.mp4',
    'ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part2.mp4',
    'ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part3.mp4',
    'ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part4.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part1.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part2.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part3.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part4.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part5.mp4',
    'ID05_SuitsVol1-1_id5_MEG_DATAPixx_part6.mp4',
    'ID06_TheBigBangTheoryVol1-1_id6_MEG_DATAPixx_part1.mp4',
    'ID06_TheBigBangTheoryVol1-1_id6_MEG_DATAPixx_part2.mp4',
    'ID07_TheBigBangTheoryVol1-2_id7_MEG_DATAPixx_part1.mp4',
    'ID07_TheBigBangTheoryVol1-2_id7_MEG_DATAPixx_part2.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part1.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part2.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part3.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part4.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part5.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part6.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part7.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part8.mp4',
    'ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part9.mp4',
    'ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part1.mp4',
    'ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part2.mp4',
    'ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part3.mp4',
    'ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part4.mp4',
    'ID10_GhostInTheShellVol1-1_id10_MEG_DATAPixx_part1.mp4',
    'ID10_GhostInTheShellVol1-1_id10_MEG_DATAPixx_part2.mp4'
]

for filename in filename_list:
    save_dir = os.path.join(save_root, filename.split('.')[0])
    os.makedirs(save_dir, exist_ok=True)
    video_path = os.path.join(file_root, filename)
    vc = VideoController(video_path)
    print('video name: ',  filename)
    num_frames = vc.frame_num
    print('{} frames'.format(num_frames))
    for i in range(num_frames):
        frame = vc.get_frame(i)
        cv2.imwrite(os.path.join(save_dir, '{}.png'.format(i)), frame)

    print('save images in ', save_dir)
