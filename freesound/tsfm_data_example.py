import config
import utils

import os
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import librosa


def get_wavlist(config_):
    train_dir = config_['train_dir']['data']
    test_dir = config_['test_dir']['data']

    train_wav = sorted(os.listdir(train_dir))
    test_wav = sorted(os.listdir(test_dir))

    print(f'Train wav files: {len(train_wav)}')
    print(f'Test wav files: {len(test_wav)}')

    train_df = pd.DataFrame({'fname': train_wav})
    test_df = pd.DataFrame({'fname': test_wav})

    train_df['train0/test1'] = pd.Series([0] * len(train_wav))
    test_df['train0/test1'] = pd.Series([0] * len(test_wav))

    df = train_df.append(test_df)
    df.set_index('fname', inplace=True)

    path = os.path.join(os.path.dirname(train_dir), 'wavlist.csv')
    df.to_csv(path)

    print(f'Wav list saved, total: {len(df)}')

    return path


def tsfm_wave(row, config_):
    item = row[1]
    pname = os.path.join('./data_wave', os.path.splitext(item['fname'])[0] + '.pkl')
    if item['train0/test1'] == 0:
        fpath = os.path.join(config_['train_dir']['data'], item['fname'])
    if item['train0/test1'] == 1:
        fpath = os.path.join(config_['test_dir']['data'], item['fname'])

    sr = config_['sampling_rate']
    if not os.path.exists(fpath):
        return
    data, _ = librosa.core.load(fpath, sr=sr, res_type='kaiser_best')
    utils.save_data(pname, data)


def tsfm_logmel(row, config_):
    item = row[1]
    pname = os.path.join('./data_logmel_delta', os.path.splitext(item['fname'])[0] + '.pkl')
    if item['train0/test1'] == 0:
        fpath = os.path.join(config_['train_dir']['data'], item['fname'])
    if item['train0/test1'] == 1:
        fpath = os.path.join(config_['test_dir']['data'], item['fname'])

    sr = config_['sampling_rate']
    if not os.path.exists(fpath):
        return
    data, sr = librosa.load(fpath, sr)

    # some audio file is empty, fill logmel with 0.
    if not len(data):
        print(f'Empty file: {fpath}')
        logmel = np.zeros((config_['n_mels'], 150))
        features = np.stack([logmel] * 3)
    else:
        melspec = librosa.feature.melspectrogram(data, sr, n_fft=config_['n_fft'], hop_length=config_['hop_length'],
                                                 n_mels=config_['n_mels'])
        logmel = librosa.core.power_to_db(melspec)
        delta = librosa.feature.delta(melspec)
        accelerate = librosa.feature.delta(logmel, order=2)
        features = np.stack((logmel, delta, accelerate))  # (3, 64, xx)

    utils.save_data(pname, features)


def tsfm_mfcc(row, config_):
    item = row[1]
    pname = os.path.join('./data_mfcc', os.path.splitext(item['fname'])[0] + '.pkl')
    if item['train0/test1'] == 0:
        fpath = os.path.join(config_['train_dir']['data'], item['fname'])
    if item['train0/test1'] == 1:
        fpath = os.path.join(config_['test_dir']['data'], item['fname'])

    sr = config_['sampling_rate']
    if not os.path.exists(fpath):
        return
    data, sr = librosa.load(fpath, sr)

    # some audio file is empty, fill logmel with 0.
    if not len(data):
        print(f'Empty file: {fpath}')
        mfcc = np.zeros((config_['n_mels'], 150))
        features = np.stack([mfcc] * 3)
    else:
        mfcc = librosa.feature.mfcc(data, sr, n_fft=config_['n_fft'], hop_length=config_['hop_length'],
                                                 n_mfcc=config_['n_mels'])
        delta = librosa.feature.delta(mfcc)
        accelerate = librosa.feature.delta(mfcc, order=2)
        features = np.stack((mfcc, delta, accelerate))  # (3, 64, xx)

    utils.save_data(pname, features)


def wav_to(wavelist: str, config_):
    df = pd.read_csv(wavelist)
    pool = Pool(10)

    # function for feature extraction
    tsfm = {'wave': tsfm_wave,
            'logmel': tsfm_logmel,
            'mfcc': tsfm_mfcc}

    tsfm = tsfm.get(config_['data_transform']['to'])
    tsfm = partial(tsfm, config_=config_)
    pool.map(tsfm, df.iterrows())


if __name__ == '__main__':
    utils.make_dirs()
    config_ = config.Config()  # make changes in params
    path = get_wavlist(config_)
    wav_to(path, config_)