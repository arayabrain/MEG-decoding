# Reimplementation of speech decoding paper by MetaAI

Paper: https://arxiv.org/pdf/2208.12266.pdf

<div align="center"><img src="assets/overview_meta2022.png" width=300></div>

## Status

Works for Gwilliams2022 dataset and Brennan2018 dataset.

## TODOs

- [ ] Full reproducibility support. Will be useful for HP tuning.
- [ ] Match accuracy to numbers reported in the paper.
- [ ] Work with huge memory consumption issue in Gwilliams multiprocessing

# Usage

## For EEG (Brennan et al. 2022)
Run `python train.py dataset=Brennan2018 rebuild_datasets=True`.
When `rebuild_datasets=False`, existing pre-processed M/EEG and pre-computing embeddings are used. This is useful if you want to run the model on exactly the same data and embeddings several times. Otherwise, the both audio embeddings are pre-computed and M/EEG data are pre-processed before training begins.

## For MEG (Gwilliams et al. 2022)

Run `python train.py dataset=Gwilliams2022 rebuild_datasets=True`
When `rebuild_datasets=False`, existing pre-processed M/EEG and pre-computing embeddings are used. This is useful if you want to run the model on exactly the same data and embeddings several times. It takes ~30 minutes to pre-process Gwilliams2022 and compute embeddings on 20 cores. Set `rebuild_datasets=False` for subsequent runs (or don't specify it, becuase by default `rebuild_datasets=False`). Otherwise, the both audio embeddings are pre-computed and M/EEG data are pre-processed before training begins.

## Monitoring training progress with W&B

To do that, set `entity` and `project` in the `wandb` section of `config.yaml`.

## Datasets

**Gwilliams et al., 2022**

- Paper https://arxiv.org/abs/2208.11488

- Dataset https://osf.io/ag3kj/

**Brennan et al., 2019**

- Paper https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0207741

- Dataset https://deepblue.lib.umich.edu/data/concern/data_sets/bg257f92t

You will need `S01.mat` to `S49.mat` placed under `data/Brennan2018/raw/` and `audio.zip` unzipped to `data/Brennan2018/audio/` to run the code.


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
