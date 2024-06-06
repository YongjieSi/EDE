# Fully Few-shot Class-incremental Audio Classification Using Expandable Dual-embedding Extractor

This repository contains the introductions to the datasets and codes used in our paper, titled "Fully Few-shot Class-incremental Audio Classification Using Expandable Dual-embedding Extractor" (as shown in the section of Citation).

## Datasets

To study the Fully Few-shot Class-incremental Audio Classification (FFCAC) problem, three datasets of LS-100 dataset, NSynth-100 dataset and FSC-89 dataset are constructed by 
choosing samples from audio corpora of the [Librispeech](https://www.openslr.org/12/) dataset, the [NSynth](https://magenta.tensorflow.org/datasets/nsynth) dataset and the [FSD-MIX-CLIPS](https://zenodo.org/record/5574135#.YWyINEbMIWo) dataset respectively.

Wei Xie, one of our team members, constructed the NSynth-100 dataset and FSC-89 dataset. The detailed information of these two datasets is [here](https://github.com/chester-w-xie/FCAC_datasets).

The detailed information of the LS-100 dataset is given below.

### Preparation of the LS-100 dataset

LibriSpeech is a corpus of approximately 1000 hours of 16kHz read English speech, prepared by Vassil Panayotov with the assistance of Daniel Povey. The data is derived from read audiobooks from the LibriVox project, and has been carefully segmented and aligned. We find that the subset ``train-clean-100`` of   Librispeech is enough for our study, so we constructed the LS-100 dataset using partial samples from the Librispeech as the source materials. To be specific, we first concatenate all the speakers' speech clips into a long speech, and then select the 100 speakers with the longest duration to cut their voices into two second speech. You can download the Librispeech from [here](https://www.openslr.org/12/).

1. Download [dataset](https://www.openslr.org/resources/12/train-clean-100.tar.gz) and extract the files.
   
2. Transfer the format of audio files. Move the script ``normalize-resample.sh`` to the root dirctory of extracted folder, and run the command ``bash normalize-resample.sh``.

3. Construct LS-100 dataset.
   
   ```
   python data/LS100/construct_LS100.py --data_dir DATA_DIR --duration_json data/librispeech/spk_total_duration.json --single_spk_dir SINGLE_SPK_DIR --num_select_spk 100 --spk_segment_dir SPK_SEGMENT_DIR --csv_path CSV_PATH --spk_mapping_path SPK_MAPPING_PATH
   ```

## 

##### Preparation of the NSynth-100 dataset

The NSynth dataset is an audio dataset containing 306,043 musical notes, each with a unique pitch, timbre, and envelope. 
Those musical notes are belonging to 1,006 musical instruments.

Based on the statistical results, we obtain the NSynth-100 dataset by the following steps:

1. Download [Train set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz), [Valid set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz), and [test set](http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz) of the NSynth dataset to your local machine and unzip them.

2. Download the meta files for FCAC from [here](./data/nsynth) to your local machine and unzip them.

3. You will get s structure of the directory as follows:

   ```
   Your dataset root(Nsynth)
   ├── nsynth-100-fs-meta
   ├── nsynth-200-fs-meta
   ├── nsynth-300-fs-meta
   ├── nsynth-400-fs-meta
   ├── nsynth-test
   │   └── audio
   ├── nsynth-train
   │   └── audio
   └── nsynth-valid
       └── audio
   ```

## Code

- run 

    ```bash
    python main.py --config=.exps/args.json
    ```


## Contact
Yanxiong Li (eeyxli@scut.edu.cn) and Yongjie Si (eeyongjiesi@mail.scut.edu.cn)
School of Electronic and Information Engineering, South China University of Technology, Guangzhou, China

## Citation
Please cite our paper if you find the codes and datasets are useful for your research.

[1] Yanxiong Li, Wenchang Cao, Jialong Li, Wei Xie, and Qianhua He, "Fully Few-shot Class-incremental Audio Classification Using Expandable Dual-embedding Extractor," in Proc. of INTERSPEECH, Kos, Greece, 01-05 Sep., 2024.


