# SSL

## pretraining approach
### W/O label
1. [Time-Frequency Consistency](https://arxiv.org/pdf/2206.08496.pdf)
2. [reconstruction by ViT](https://arxiv.org/pdf/2306.16934.pdf)

### W/ label
1. BLIP
2. CLIP

## GOD
/storage/dataset/image/ImageNet/ILSVRC2012_val/

## MEG-Movie (yanagisawa-lab)
### video name
動画は10タイトルで、各タイトル複数パートに分割して提示した(40 sessions)。その際分割された動画の長さは15分以内に収まるようになっている。
- その他，ビデオに関するメモ
  ID10_GhostInTheShellVol1-1だけビデオが25fpsで，その他のビデオは29.97fpsである．
  エンコードパラメータはすべて，映像: 6.2Mbps; 音声: 48kHz，224kbps で統一している  29.9

```
ID01_HerosVol1-1_id1_MEG_DATAPixx_part1.mp4         ID05_SuitsVol1-1_id5_MEG_DATAPixx_part6.mp4
ID01_HerosVol1-1_id1_MEG_DATAPixx_part2.mp4         ID06_TheBigBangTheoryVol1-1_id6_MEG_DATAPixx_part1.mp4
ID01_HerosVol1-1_id1_MEG_DATAPixx_part3.mp4         ID06_TheBigBangTheoryVol1-1_id6_MEG_DATAPixx_part2.mp4
ID01_HerosVol1-1_id1_MEG_DATAPixx_part4.mp4         ID07_TheBigBangTheoryVol1-2_id7_MEG_DATAPixx_part1.mp4
ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part1.mp4  ID07_TheBigBangTheoryVol1-2_id7_MEG_DATAPixx_part2.mp4
ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part2.mp4  ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part1.mp4
ID02_TheMentalistVol1-1_id2_MEG_DATAPixx_part3.mp4  ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part2.mp4
ID03_GleeVol1-1_id3_MEG_DATAPixx_part1.mp4          ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part3.mp4
ID03_GleeVol1-1_id3_MEG_DATAPixx_part2.mp4          ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part4.mp4
ID03_GleeVol1-1_id3_MEG_DATAPixx_part3.mp4          ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part5.mp4
ID03_GleeVol1-1_id3_MEG_DATAPixx_part4.mp4          ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part6.mp4
ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part1.mp4      ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part7.mp4
ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part2.mp4      ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part8.mp4
ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part3.mp4      ID08_DreamGirlsVol1-1_id8_MEG_DATAPixx_part9.mp4
ID04_TheCrownVol1-1_id4_MEG_DATAPixx_part4.mp4      ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part1.mp4
ID05_SuitsVol1-1_id5_MEG_DATAPixx_part1.mp4         ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part2.mp4
ID05_SuitsVol1-1_id5_MEG_DATAPixx_part2.mp4         ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part3.mp4
ID05_SuitsVol1-1_id5_MEG_DATAPixx_part3.mp4         ID09_BreakingBadVol1-1_id9_MEG_DATAPixx_part4.mp4
ID05_SuitsVol1-1_id5_MEG_DATAPixx_part4.mp4         ID10_GhostInTheShellVol1-1_id10_MEG_DATAPixx_part1.mp4
ID05_SuitsVol1-1_id5_MEG_DATAPixx_part5.mp4         ID10_GhostInTheShellVol1-1_id10_MEG_DATAPixx_part2.mp4
```
### データ注意事項
1. **sbj01**
    * videoID=10, partID=1,2に対応するMEGデータには、ディジタルトリガーが入っていないため、光トリガーで代用する。

2. **others**

### 仕様
* **MEG-only mode**
    * 各Sessionごとに、事前にMEGの前処理を行う
    * MEGデータ（.con）と動画データ(.mp4)から動画のフレーム提示を指示するtriggerを取り出す。このトリガーは仕様上動画2frameごとに一回立ち上がる。
        * .conファイルのパースはMATLABでしかできないためこのリポジトリには含まれていない。
    * 各sessionをセグメント（60sec）に分割する
    * train/valにセグメントを分割。
    * 間隔TごとにMEGデータを分割
* **MEG-Drama pair mode**


### データ条件
以下の項目の積を想定
1. サイズ
2. 被験者混合 / 非混合
3. 外部データ（視覚以外のタスク）混合/非混合
4. source-reconstruction
    1. Early Visual
    2. ventral
    3. MT+_complex
    4. Dorsal visual
    5. Auditory
    6. Early Auditory
    7. Auditory Associsation


### コマンド履歴
#### trigger creation
Use MATLAB:  
extract_triggers2.m


`python train_ssl.py --config test_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config diffusion_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config diffusion_16_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`
#### sbj01-all (31138 trials)
`train_ssl.py --config sbj1_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj1_all --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500 --h5name vc-fs1000-dura200-2`

`python train_ssl.py --config sbj1_all --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200 --h5name vc-fs1000-dura200-3`
###### sbj03
`python train_ssl.py --config sbj3_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name sbj3-vc-fs1000-dura200-1  --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj3_10k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-10k --h5name sbj3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt`

python train_ssl.py --config sbj3_5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-5k --h5name sbj3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt

python train_ssl.py --config sbj3_2.5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-2.5k --h5name sbj3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt

`python train_ssl.py --config sbj3_1k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-1k --h5name sbj3-vc-fs1000-dura200-3  --wandbkey /home/yainoue/wandb_inoue.txt`


##### sbj01-03
python train_ssl.py --config sbj1_3_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name sbj1_3-vc-fs1000-dura200-1  --wandbkey /home/yainoue/wandb_inoue.txt

python train_ssl.py --config sbj1_3_10k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-10k --h5name sbj1_3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt

python train_ssl.py --config sbj1_3_5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-5k --h5name sbj1_3-vc-fs1000-dura200-1  --wandbkey /home/yainoue/wandb_inoue.txt

python train_ssl.py --config sbj1_3_2.5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-2.5k --h5name sbj1_3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt

python train_ssl.py --config sbj1_3_1k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-1k --h5name sbj1_3-vc-fs1000-dura200-2  --wandbkey /home/yainoue/wandb_inoue.txt
<!-- 
`python train_ssl.py --config sbj3_all --model scmbm --preprocess fs1000_dura500 --device_counts 1 --exp scmbm_4-fs1000-dura500 --h5name sbj3-vc-fs1000-dura200-2 --wandbkey /home/yainoue/wandb_inoue.txt  `

`python train_ssl.py --config sbj3_all --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1  --exp scmbm_16-fs1000-dura200 --h5name sbj3-vc-fs1000-dura200-3  --wandbkey /home/yainoue/wandb_inoue.txt` -->

#### sbj01-10k

`python train_ssl.py --config sbj1_10k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-10k --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj1_10k --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-10k --h5name vc-fs1000-dura200-2`

`python train_ssl.py --config sbj1_10k --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200-10k --h5name vc-fs1000-dura200-3`

#### sbj01-5k

`python train_ssl.py --config sbj1_5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-5k --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj1_5k --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-5k --h5name vc-fs1000-dura200-2`

`python train_ssl.py --config sbj1_5k --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200-5k --h5name vc-fs1000-dura200-3`

#### sbj01-1k

`python train_ssl.py --config sbj1_1k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-1k --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj1_1k --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-1k --h5name vc-fs1000-dura200-2`

`python train_ssl.py --config sbj1_1k --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200-1k --h5name vc-fs1000-dura200-3`

#### sbj01-2.5k

`python train_ssl.py --config sbj1_2.5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-2.5k --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python train_ssl.py --config sbj1_2.5k --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-2.5k --h5name vc-fs1000-dura200-2`

`python train_ssl.py --config sbj1_2.5k --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200-2.5k --h5name vc-fs1000-dura200-3`

###　下位タスク(Alignment)  
####  sbj01-all (31138 trials)  

`python contrastive_learning.py --config sbj1_6k --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`python contrastive_learning.py --config sbj1_6k --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura500 --device_counts 1 --exp scmbm_4-fs1000-dura500 --h5name vc-fs1000-dura200-2 --wandbkey /home/yainoue/wandb_inoue.txt`

`python contrastive_learning.py --config sbj1_6k --meg_model scmbm_16 --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_16-fs1000-dura200 --h5name vc-fs1000-dura200-3 --wandbkey /home/yainoue/wandb_inoue.txt`
==============================================
cossim
`python contrastive_learning.py --config sbj1_6k_cos_sim --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`
cossim-global pool
`python contrastive_learning.py --config sbj1_6k_cos_sim_gp --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-2 --wandbkey /home/yainoue/wandb_inoue.txt`
clip
`python contrastive_learning.py --config sbj1_6k_clip --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-3 --wandbkey /home/yainoue/wandb_inoue.txt`
cossim-global pool lora
`python contrastive_learning.py --config sbj1_6k_cos_sim_gp_lora --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-2 --wandbkey /home/yainoue/wandb_inoue.txt`

# TODO
`python contrastive_learning.py --config sbj1_6k_cos_sim --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura500 --device_counts 1 --exp scmbm_4-fs1000-dura500 --h5name vc-fs1000-dura200-2 --wandbkey /home/yainoue/wandb_inoue.txt`

`python contrastive_learning.py --config sbj1_6k_cos_sim --meg_model scmbm_16 --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_16-fs1000-dura200 --h5name vc-fs1000-dura200-3 --wandbkey /home/yainoue/wandb_inoue.txt`

# 0906 Downstream
`python contrastive_learning.py --config sbj1_god_drama_clip --meg_model scmbm --vision_model vit_clip16 --decode_model mlp --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`


## Diffusion
`python eeg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1`

## Diffusion2
`python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1`  

python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test   

## Diffusion3(only generator)
python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-1k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen-1k --datadir sbj1

python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-5k --meg_h5name vc-fs1000-dura200-3 --device_counts 1 --ldf_exp test_gen-5k --datadir sbj1

python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-10k --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_gen-10k --datadir sbj1

python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-2.5k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen-2.5k --datadir sbj1


## Diffusion3(debug)
python meg_ldm.py --dataset MEG  --num_epoch 300 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-1k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp debug --datadir sbj1

## Diffusin 4
python meg_ldm.py --dataset MEG  --num_epoch 100 --batch_size 4  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_gen_enc --datadir sbj1

python meg_ldm_generate.py --dataset MEG  --num_epoch 100 --batch_size 4  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_gen_enc --datadir sbj1

nohup python meg_ldm.py --dataset MEG  --num_epoch 500 --batch_size 2  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_enc-6k --datadir sbj1 > logs/0914_diffusion.out &

nohup python meg_ldm.py --dataset MEG  --num_epoch 500 --batch_size 2  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_enc-6k-resume --datadir sbj1 > logs/0914_diffusion.out &

nohup python meg_ldm.py --dataset MEG  --num_epoch 500 --batch_size 2  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-5k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_enc-6k-resume --datadir sbj1 > logs/0922_diffusion.out &

nohup python meg_ldm.py --dataset MEG  --num_epoch 500 --batch_size 2  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-1k --meg_h5name vc-fs1000-dura200-3 --device_counts 1 --ldf_exp test_enc-6k-resume --datadir sbj1 > logs/0922_diffusion1k.out &

### only generator
#### MEG Encoder all
python meg_ldm.py --dataset MEG  --num_epoch 100 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen 
#### MEG Encoder 1k
python meg_ldm.py --dataset MEG  --num_epoch 100 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-1k --meg_h5name vc-fs1000-dura200-1 --device_counts 1 --ldf_exp test_gen-1k 

#### MEG Encoder 5k
python meg_ldm.py --dataset MEG  --num_epoch 100 --batch_size 4  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-5k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen-5k 

#### evaluate
python meg_ldm_generate.py --dataset MEG  --meg_config config_generative_model_test_only_generator --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200-5k --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen-5k 


python meg_ldm_generate.py --dataset MEG  --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-2 --device_counts 1 --ldf_exp test_gen_enc

python meg_ldm_generate.py --dataset MEG --meg_config config_generative_model_test --meg_encoder scmbm --meg_preprocess fs1000_dura200 --meg_exp scmbm_4-fs1000-dura200 --meg_h5name vc-fs1000-dura200-1 --ldf_exp test_enc-6k

## ROI=ALL
`train_ssl.py --config sbj1_all_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_all --h5name fs1000-dura200-roi_all-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_10k_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_all-10k --h5name fs1000-dura200-roi_all-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_2.5k_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_all-2.5k --h5name fs1000-dura200-roi_all-2 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_1k_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_all-1k --h5name fs1000-dura200-roi_all-3 --wandbkey /home/yainoue/wandb_inoue.txt`

## ROI=frontal
`train_ssl.py --config sbj1_frontal_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_frontal --h5name fs1000-dura200-roi_frontal-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_frontal_10k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_frontal-10k --h5name fs1000-dura200-roi_frontal-2 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_frontal_5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_frontal-5k --h5name fs1000-dura200-roi_frontal-1 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_frontal_2.5k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_frontal-2.5k --h5name fs1000-dura200-roi_frontal-2 --wandbkey /home/yainoue/wandb_inoue.txt`

`train_ssl.py --config sbj1_frontal_1k --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200-roi_frontal-1k --h5name fs1000-dura200-roi_frontal-2 --wandbkey /home/yainoue/wandb_inoue.txt`

## SEED change
`python train_ssl.py --config sbj1_10k --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-10k-seed_0 --h5name vc-fs1000-dura200-3 --seed 0`

`python train_ssl.py --config sbj1_all --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500-seed_0 --h5name vc-fs1000-dura200-2 --seed 0`

### 実験計画
#### 事前学習
1. 訓練データサイズ
    * 100, 1000, 5000, 10000, 100000
2. patchサイズ
    * 4
    * 16
3. resample rate
    * 1000 Hzが良さそう
4. duration
    * 0.2 s
    * 0.5 s
5. ROI
    * VC
    * all
6. reconstruction
    * False
    * True
7. Mixing subject
    * False
    * True

#### 下位タスク
1. Approach
    * contrastive learning (cosine-similarity)
    * MSE
    * cosin-sim
    * latent-diffusion
2. fine-tuning 手法
    * freeze
    * LoRA
    * full-fine-tuning
3. Label
    * w/o labels of pretraining data
    * w/ labels of pretraining data


