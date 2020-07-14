""" Make submission """

from v2.config import DefaultConfig

from scipy.stats import gmean
import numpy as np
import pandas as pd


def calibrate(p, n_train: int, n_classes: int, c_classes: pd.DataFrame):

    def calibrate_p(p_train, p_test, p):
        return p_test * p / (p_test * p + p_train * (1 - p))

    p_new = p.copy()
    for c in range(n_classes):
        p_train = c_classes.loc[c] / n_train
        p_test = 1 / n_classes
        for i in range(len(p)):
            p_new[i, c] = calibrate_p(p_train, p_test, p[i, c])
    return p_new


def make_preds(fnames, p_calibrate=True):
    train = pd.read_csv(conf.rawdata.train['csv'])
    test = pd.read_csv(conf.rawdata.test['csv'])

    # ignore the empty wavs
    test['remove'] = 0
    f_empty = ['b39975f5.wav', '6ea0099f.wav', '0b0427e2.wav']
    test.loc[test.fname.isin(f_empty), 'remove'] = 1

    labels = conf.rawdata.labels[:]  # make copy
    p = gmean(np.stack([np.load(fn) for fn in fnames], axis=2), axis=2)

    # probabilities calibration
    if p_calibrate:
        labels_idx = {label: i for i, label in enumerate(labels)}
        train['label_idx'] = train.label.apply(lambda x: labels_idx[x])
        p = calibrate(p, len(train), len(labels), train.label_idx.value_counts())

    top_3 = np.array(labels)[np.argsort(-p, axis=1)[:, :3]]
    p_labels = [' '.join(list(x)) for x in top_3]

    inds = test[test.remove == 0].index
    test.at[inds, 'label'] = np.array(p_labels)

    test.set_index('fname', inplace=True)
    test[['label']].to_csv('sbm.csv')
    print(f'Result saved as sbm.csv')


if __name__ == '__main__':
    conf = DefaultConfig()
    fnames = [
        # Resnext preds
        './tb_logs/resnext101_32x4d/fold_1/preds.npy',
        './tb_logs/resnext101_32x4d/fold_1/preds_v1.npy', './tb_logs/resnext101_32x4d/fold_1/preds_v2.npy',
        './tb_logs/resnext101_32x4d/fold_1/preds_v3.npy', './tb_logs/resnext101_32x4d/fold_1/preds_v4.npy',

        './tb_logs/resnext101_32x4d/fold_1_r/preds.npy',
        './tb_logs/resnext101_32x4d/fold_1_r/preds_v1.npy', './tb_logs/resnext101_32x4d/fold_1_r/preds_v2.npy',
        './tb_logs/resnext101_32x4d/fold_1_r/preds_v3.npy', './tb_logs/resnext101_32x4d/fold_1_r/preds_v4.npy',

        # Resnet50 augmented preds
        './tb_logs/resnet50/fold_1/preds.npy',
        './tb_logs/resnet50/fold_1/preds_v1.npy', './tb_logs/resnet50/fold_1/preds_v2.npy',
        './tb_logs/resnet50/fold_1/preds_v3.npy', './tb_logs/resnet50/fold_1/preds_v4.npy',

        './tb_logs/resnet50/fold_1_r/preds.npy',
        './tb_logs/resnet50/fold_1_r/preds_v1.npy', './tb_logs/resnet50/fold_1_r/preds_v2.npy',
        './tb_logs/resnet50/fold_1_r/preds_v3.npy', './tb_logs/resnet50/fold_1_r/preds_v4.npy',
    ]
    make_preds(fnames)