""" Make 5 folds for training """

from sklearn.model_selection import StratifiedKFold


def make_folds(df, n=5, seed=33):
    kfold = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    folds = list(kfold.split(df.fname.values, df.label.values))

    for train_idx, valid_idx in folds:
        yield df.iloc[train_idx].reset_index(drop=True), df.iloc[valid_idx].reset_index(drop=True)


