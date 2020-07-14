""" Compute Log Mel-Spectrograms """

from v2.config import DefaultConfig

import os
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
import numpy as np
import librosa


def logmel(fn, from_, to_):
    wav = np.load(os.path.join(from_, fn + '.npy'))
    wav = librosa.resample(wav, conf.rawdata.sampling_rate, conf.logmel.sampling_rate)
    melspec = librosa.feature.melspectrogram(wav,
                                             sr=conf.logmel.sampling_rate,
                                             n_fft=conf.logmel.n_fft,
                                             hop_length=conf.logmel.hop_length,
                                             n_mels=conf.logmel.n_mels)

    mode = 'interp' if melspec.shape[1] >= 9 else 'mirror'
    logmel = librosa.core.power_to_db(melspec)
    delta = librosa.feature.delta(melspec, mode=mode)
    accelerate = librosa.feature.delta(logmel, order=2, mode=mode)
    features = np.stack((logmel, delta, accelerate))

    np.save(os.path.join(to_, fn + '.npy'), features)


def wave(fn, from_, to_):
    raise NotImplemented


def mfcc(fn, from_, to_):
    raise NotImplemented


def process(fnames, dir_name):
    method = {
        'logmel': {'func': logmel, 'dir_prefix': '_logmel'},
        'mfcc': None,
        'wave': None
    }
    kwargs = method[conf.method]

    new_dir_name = dir_name + kwargs['dir_prefix']
    os.makedirs(new_dir_name, exist_ok=True)
    print(f'New dir, {new_dir_name}')

    pool = Pool(processes=10)
    func = partial(kwargs['func'], from_=dir_name + '_trim', to_=new_dir_name)
    pool.map(func, tqdm(fnames))


def process_data():
    train = pd.read_csv(conf.rawdata.train['csv'])
    test = pd.read_csv(conf.rawdata.test['csv'])

    # ignore the empty wavs
    test['remove'] = 0
    f_empty = ['b39975f5.wav', '6ea0099f.wav', '0b0427e2.wav']
    test.loc[test.fname.isin(f_empty), 'remove'] = 1

    print('Train ...')
    process(train.fname.values, conf.rawdata.train['wav'])

    print('Test ...')
    process(test[test.remove == 0].fname.values, conf.rawdata.test['wav'])


if __name__ == '__main__':
    conf = DefaultConfig()
    process_data()