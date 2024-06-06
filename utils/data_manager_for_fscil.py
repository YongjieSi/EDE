import logging
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchaudio
import librosa
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from utils.data import filibrispeech, fiNsynth100, fiFMC89


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, shot=5):
        self.dataset_name = dataset_name
        self.shot = shot
        self.seed = seed
        self._setup_data(dataset_name, shuffle, seed) 
        assert init_cls <= len(self._class_order), 'No enough classes.'
        self._increments = [init_cls]  
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)  
        if offset > 0:  
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False):
        """
        Args:
            indices(array or list): the category you want to select
            source(str): 'train' or 'test', which kind of data want to use
            mode(str): if mode=="test", take all the data; if mode="train"
            appendent(tuple): a tuple consists of data and its label
            ret_data(bool): if true, return data and label and DummyDataset
        Returns:
            data():
            target():
            DummyDataset(Class): a class inherits from torch.utils.data.Dataset, a costome dataset class
        """
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        data, targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1) 
            if mode == "train":
                torch.manual_seed(self.seed )
                pos = (torch.randperm(len(class_data)))[:self.shot].numpy()  
                class_data, class_targets = class_data[pos], class_targets[pos]
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, self.use_path)
        else:
            return DummyDataset(data, targets, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == 'train':
            x, y = self._train_data, self._train_targets
        elif source == 'test':
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError('Unknown data source {}.'.format(source))

        if mode == 'train':
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == 'test':
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError('Unknown mode {}.'.format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx+1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets))+1):
                append_data, append_targets = self._select(appendent_data, appendent_targets,
                                                           low_range=idx, high_range=idx+1)
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return DummyDataset(train_data, train_targets, trsf, self.use_path), \
            DummyDataset(val_data, val_targets, trsf, self.use_path)

    def _setup_data(self, dataset_name, shuffle, seed):
        """
        Args:
            dataset_name(str): which dataset
            shuffle(bool): shuffle the labels of data or not, if true, the data of 
                            same category's label will map to another value simultaneously
            seed(int):
        Return:

        """
        idata = _get_idata(dataset_name) 
        idata.download_data() 

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path


        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        """
        Args:
            x: all the data(images, features or file paths)
            y: all the labels
            low_range: 
            high_range:
        Return:
            selected_x: data whose label from low_range to high_range(exclude)
            selected_y: labels from low_range to high_range(exclude)
        """
        x, y = np.asarray(x), np.asarray(y)
        # select only one class from x and y
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]


class DummyDataset(Dataset):
    def __init__(self, audios, labels, trsf=None, use_path=True):
        assert len(audios) == len(labels), 'Data size error!'
        self.audios = audios.tolist()
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        if self.use_path:
            audio, sr = librosa.load(self.audios[idx], sr=None)
        else:
            audio = self.audios[idx]
        label = self.labels[idx]

        return idx, audio, label


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "nsynth-100":
        return fiNsynth100()
    elif name == "fmc":
        return fiFMC89()
    elif name == "librispeech":
        return filibrispeech()
    else:
        raise NotImplementedError('Unknown dataset {}.'.format(dataset_name))


if __name__ == "__main__":
    data_manager = DataManager("Nsynth-100", False, 1993, 60, 5)
    nb_tsks = data_manager.nb_tasks
    task_size = data_manager.get_task_size(0)
    train_dataset = data_manager.get_dataset(np.arange(60, 100), source='train',
                                                mode='train')
    test_dataset = data_manager.get_dataset(np.arange(0, 100), source='test', mode='test')