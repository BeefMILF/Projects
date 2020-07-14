""" Trim data and trailing silence """

from v2.config import DefaultConfig


import os
from functools import partial
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
from multiprocessing import Pool


def trim(fn, sr, from_, to_):
    x, _ = librosa.load(os.path.join(from_, fn), sr)
    x = librosa.effects.trim(x)[0]
    np.save(os.path.join(to_, fn + '.npy'), x)


def process(fnames, dir_name, sr):
    new_dir_name = dir_name + '_trim'
    os.makedirs(new_dir_name, exist_ok=True)
    print(f'New dir, {new_dir_name}')

    pool = Pool(processes=10)
    func = partial(trim, sr=sr, from_=dir_name, to_=new_dir_name)
    pool.map(func, tqdm(fnames))


def trim_data():
    train = pd.read_csv(conf.train['csv'])
    test = pd.read_csv(conf.test['csv'])

    # ignore the empty wavs
    test['remove'] = 0
    f_empty = ['b39975f5.wav', '6ea0099f.wav', '0b0427e2.wav']
    test.loc[test.fname.isin(f_empty), 'remove'] = 1

    print('Train ...')
    process(train.fname.values, conf.train['wav'], conf.sampling_rate)

    print('Test ...')
    process(test[test.remove == 0].fname.values, conf.test['wav'], conf.sampling_rate)


if __name__ == '__main__':
    conf = DefaultConfig().rawdata
    trim_data()
