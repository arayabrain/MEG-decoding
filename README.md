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

`python train_ssl.py --config test_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`  
`python train_ssl.py --config diffusion_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`  
`python train_ssl.py --config diffusion_16_config  --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt`  

`train_ssl.py --config sbj1_all --model scmbm --preprocess fs1000_dura200 --device_counts 1 --exp scmbm_4-fs1000-dura200 --h5name vc-fs1000-dura200-1 --wandbkey /home/yainoue/wandb_inoue.txt`  

`python train_ssl.py --config sbj1_all --model scmbm --preprocess fs1000_dura500 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_4-fs1000-dura500 --h5name vc-fs1000-dura200-2`     

`python train_ssl.py --config sbj1_all --model scmbm_16 --preprocess fs1000_dura200 --device_counts 1 --wandbkey /home/yainoue/wandb_inoue.txt  --exp scmbm_16-fs1000-dura200 --h5name vc-fs1000-dura200-3`   
  

### 実験計画
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