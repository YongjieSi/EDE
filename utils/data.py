import os
import json
import numpy as np
from torchvision import datasets, transforms
from utils.toolkit import split_images_labels
import pandas as pd
#### the Class of all i* dataset return all the train data with target and test data with target

class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None



class fiNsynth100(iData):
    use_path = True

    class_order = np.arange(25).tolist()

    def download_data(self):
        root_dir = '/data/datasets/The_NSynth_Dataset/'

        all_train_df = pd.read_csv("/data/syj/PyCIL/data/nsynth/nsynth-100-fs-meta/nsynth-100-fs_train.csv")
        all_val_df = pd.read_csv("/data/syj/PyCIL/data/nsynth/nsynth-100-fs-meta/nsynth-100-fs_val.csv")
        all_test_df = pd.read_csv("/data/syj/PyCIL/data/nsynth/nsynth-100-fs-meta/nsynth-100-fs_test.csv")
        with open("/data/syj/PyCIL/data/nsynth/nsynth-100-fs-meta/nsynth-100-fs_vocab.json") as vocab_json_file:
            label_to_ix = json.load(vocab_json_file)
        self.train_data = [os.path.join(root_dir, all_train_df['audio_source'][i], 'audio', all_train_df['filename'][i] + '.wav') \
                                            for i in range(len(self.class_order)*200)]
        self.train_targets = [label_to_ix[all_train_df['instrument'][i]] for i in range(len(self.class_order)*200)]

        self.test_data = [os.path.join(root_dir, all_test_df['audio_source'][i], 'audio', all_test_df['filename'][i] + '.wav') \
                                            for i in range(len(self.class_order)*100)]
        self.test_targets = [label_to_ix[all_test_df['instrument'][i]] for i in range(len(self.class_order)*100)]


class fiFMC89(iData):
    use_path = True

    class_order = np.arange(25).tolist()

    def download_data(self):
        root_dir = '/data/datasets/FSD-MIX-CLIPS-for_FSCIL/FSD-MIX-CLIPS_data'

        all_train_df = pd.read_csv("/data/syj/PyCIL/data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_train.csv")
        all_val_df = pd.read_csv("/data/syj/PyCIL/data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_val.csv")
        all_test_df = pd.read_csv("/data/syj/PyCIL/data/FMC/FSD_MIX_CLIPS.annotations_revised/FSC-89-meta/mini/Fsc89-mini-fsci_test.csv")
        self.train_data = [os.path.join(root_dir, "audio", all_train_df['data_folder'][i], \
                        all_train_df['FSD_MIX_SED_filename'][i].replace('.wav', '_' + str(int(all_train_df['start_time'][i] * 44100)) + '.wav')) \
                                            for i in range(len(self.class_order)*800)]
        self.train_targets =[all_train_df['label'][i] for i in range(len(self.class_order)*800)]

        self.test_data = [os.path.join(root_dir, "audio", all_test_df['data_folder'][i], \
                        all_test_df['FSD_MIX_SED_filename'][i].replace('.wav', '_' + str(int(all_test_df['start_time'][i] * 44100)) + '.wav')) \
                                            for i in range(len(self.class_order)*200)]
        self.test_targets = [all_test_df['label'][i] for i in range(len(self.class_order)*200)]


class filibrispeech(iData):
    use_path = True

    class_order = np.arange(25).tolist()

    def download_data(self):
        root_dir = '/data/datasets/librispeech_fscil'

        all_train_df = pd.read_csv("/data/syj/PyCIL/data/librispeech/librispeech_fscil_train.csv")
        all_val_df = pd.read_csv("/data/syj/PyCIL/data/librispeech/librispeech_fscil_val.csv")
        all_test_df = pd.read_csv("/data/syj/PyCIL/data/librispeech/librispeech_fscil_test.csv")

        self.train_data = [os.path.join(root_dir, "100spks_segments", all_train_df['filename'][i]) \
                                            for i in range(len(self.class_order)*500)]
        self.train_targets = [all_train_df['label'][i] for i in range(len(self.class_order)*500)]

        self.test_data = [os.path.join(root_dir, "100spks_segments", all_test_df['filename'][i]) \
                                            for i in range(len(self.class_order)*100)]
        self.test_targets = [all_test_df['label'][i] for i in range(len(self.class_order)*100)]