""" Make datasets, loaders for training, validation, test"""

from v2.config import DefaultConfig
from v2.stratify import make_folds
from v2.misc import transform, augment

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Freesound(Dataset):
    def __init__(self, dir_name, df, mode, transform=None):
        self.dir_name = dir_name
        self.df = df
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        fn = self.df.fname[idx] + '.npy'
        fp = os.path.join(self.dir_name, fn)

        # Read and resample the audio
        data = self._random_selection(fp)

        if self.transform is not None:
            data = self.transform(data)

        if self.mode == 'train' or self.mode == 'val':
            return data, self.df.label_idx[idx]
        elif self.mode == 'test':
            return data

    def _random_selection(self, fpath):
        raise NotImplemented


class Freesound_logmel(Freesound):
    def __init__(self, conf, dir_name, df, mode, transform=None, augment=None):
        super().__init__(dir_name=dir_name, df=df, mode=mode, transform=transform)
        self.conf = conf
        self.augment = augment
        self.in_len = conf.input_length

    def __getitem__(self, idx):
        return super().__getitem__(idx)

    def _random_selection(self, fp):
        logmel = np.load(fp)
        n = logmel.shape[2]

        # Random offset / padding
        max_offset = n - self.in_len

        # Augmentation, if logmel size < in_len
        if max_offset < 0 and self.augment is not None:
            logmel = self.augment(torch.tensor(logmel))
            logmel = logmel.numpy()
            n = logmel.shape[2]
            max_offset = n - self.in_len

        if max_offset > 0:
            offset = np.random.randint(max_offset)
            logmel = logmel[:, :, offset:(self.in_len + offset)]
        else:
            if max_offset < 0:
                max_offset = abs(max_offset)
                offset = np.random.randint(max_offset)
            else:
                offset = 0
            logmel = np.pad(logmel, ((0, 0), (0, 0), (offset, max_offset - offset)), 'constant')

        return logmel


def make_dataloaders(conf: DefaultConfig, train, val=None, test=None):

    method = {
        'logmel': {'dataset': Freesound_logmel, 'dir_prefix': '_logmel'},
        'mfcc': None,
        'wave': None
    }
    kwargs = method[conf.dataloader.method]
    Dataset = kwargs['dataset']

    train_dataset = Dataset(conf.dataloader,
                            dir_name=conf.rawdata.train['wav'] + kwargs['dir_prefix'],
                            df=train,
                            mode='train',
                            transform=transform['train'][conf.dataloader.train_transform],
                            augment=augment['train'][conf.dataloader.train_aug])
    train_dataloader = DataLoader(train_dataset, batch_size=conf.dataloader.batch_size, shuffle=True, num_workers=4)

    if val is not None:
        val_dataset = Dataset(conf.dataloader,
                              dir_name=conf.rawdata.train['wav'] + kwargs['dir_prefix'],
                              df=val,
                              mode='val',
                              transform=transform['val'][conf.dataloader.val_transform],
                              augment=augment['val'][conf.dataloader.val_aug])
        val_dataloader = DataLoader(val_dataset, batch_size=conf.dataloader.batch_size, shuffle=False, num_workers=4)
    else:
        val_dataloader = None

    if test is not None:
        test_dataset = Dataset(conf.dataloader,
                               dir_name=conf.rawdata.test['wav'] + kwargs['dir_prefix'],
                               df=test,
                               mode='test',
                               transform=transform['test'][conf.dataloader.test_transform],
                               augment=augment['test'][conf.dataloader.test_aug])
        test_dataloader = DataLoader(test_dataset, batch_size=conf.dataloader.batch_size, shuffle=False, num_workers=4)
    else:
        test_dataloader = None

    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }


def stratified_loaders(conf: DefaultConfig):
    train = pd.read_csv(conf.rawdata.train['csv'])
    test = pd.read_csv(conf.rawdata.test['csv'])

    # ignore the empty wavs
    test['remove'] = 0
    f_empty = ['b39975f5.wav', '6ea0099f.wav', '0b0427e2.wav']
    test.loc[test.fname.isin(f_empty), 'remove'] = 1

    labels = conf.rawdata.labels[:]  # make copy
    labels_idx = {label: i for i, label in enumerate(labels)}

    train['label_idx'] = train.label.apply(lambda x: labels_idx[x])

    n_folds = conf.dataloader.n_folds
    if n_folds:
        generator = make_folds(train, n_folds, conf.dataloader.seed)
        for i, (train, val) in enumerate(generator, 1):
            yield make_dataloaders(conf, train, val, test[test.remove == 0].reset_index(drop=True))
    else:
        yield make_dataloaders(conf, train, val=None, test=test[test.remove == 0].reset_index(drop=True))


def test_loaders(conf: DefaultConfig, n=5):
    """ Test loader generator, used for making predictions """
    test = pd.read_csv(conf.rawdata.test['csv'])

    # ignore the empty wavs
    test['remove'] = 0
    f_empty = ['b39975f5.wav', '6ea0099f.wav', '0b0427e2.wav']
    test.loc[test.fname.isin(f_empty), 'remove'] = 1

    test = test[test.remove == 0].reset_index(drop=True)

    method = {
        'logmel': {'dataset': Freesound_logmel, 'dir_prefix': '_logmel'},
        'mfcc': None,
        'wave': None
    }
    kwargs = method[conf.dataloader.method]
    Dataset = kwargs['dataset']

    for i in range(n):
        test_dataset = Dataset(conf.dataloader,
                           dir_name=conf.rawdata.test['wav'] + kwargs['dir_prefix'],
                           df=test,
                           mode='test',
                           transform=transform['test'][conf.dataloader.test_transform])
        test_dataloader = DataLoader(test_dataset, batch_size=conf.dataloader.batch_size, shuffle=False, num_workers=4)
        yield test_dataloader


if __name__ == '__main__':
    conf = DefaultConfig()
    print(f'{conf.dataloader.n_folds} folds')
    for loaders in stratified_loaders(conf):
        print(f'Train: {len(loaders["train"])}')
        print(f'Val: {len(loaders["val"])}')
        print(f'Test: {len(loaders["test"])}')

    conf.dataloader.n_folds = 0
    print(f'{conf.dataloader.n_folds} folds')
    for loaders in stratified_loaders(conf):
        print(f'Train: {len(loaders["train"])}')
        print(f'Test: {len(loaders["test"])}')
