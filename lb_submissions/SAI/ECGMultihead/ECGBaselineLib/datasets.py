import numpy as np
import pandas as pd
import os
from pathlib import Path
    
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from pathlib import Path

# Iterative stratification
def make_stratification(df, strat_matrix, random_state):
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_split, test_split = list(msss.split(df.record_name.values[:,None], strat_matrix))[0]
    # Obtain record numbers
    train_names = df.loc[train_split, "record_name"].values
    test_names = df.loc[test_split, "record_name"].values
    # Make val/test split
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=random_state)
    val_split, test_split = list(msss.split(test_names[:,None], strat_matrix[test_split]))[0]
    assert np.intersect1d(train_names, test_names[val_split]).shape[0] == 0 & np.intersect1d(test_names[val_split], test_names[test_split]).shape[0] == 0 & np.intersect1d(test_names[test_split], train_names).shape[0] == 0, "В разбияниях повторяются записи!"
    return train_names, test_names[val_split], test_names[test_split]
    
    
##### Split for the multihead baseline ######
def get_dataset_baseline(data_path, dtype, random_state):
    assert dtype in ["train", "test"]
    classes_splits = {"ecgs":[], "targets":[], "names": []}
    metadata = pd.read_json(Path(data_path) / (dtype + "/" + dtype + ".jsonl"), lines=True)
    for signal in (Path(data_path) / dtype).glob("*.npy"):
        signal_name = signal.name[:signal.name.rfind('/')-3]
        classes_splits["names"].append(signal_name)
        with open(signal, "rb") as f:
            signal_value = np.load(f, allow_pickle=True)
        classes_splits['ecgs'].append(signal_value)
        if dtype == "train":
            signal_target = np.zeros(73)
            signal_target[metadata.loc[metadata.record_name == signal_name, "labels"].item()] = 1
            classes_splits["targets"].append(signal_target)
    classes_splits["ecgs"] = np.array(classes_splits["ecgs"])
    classes_splits["names"] = np.array(classes_splits["names"])
    if dtype == "test":
        del classes_splits["targets"]
        return classes_splits['ecgs'], classes_splits["names"]
    else:
        classes_splits["targets"] = np.array(classes_splits["targets"])
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.33, random_state=random_state)
        train_split, val_split = list(msss.split(classes_splits["ecgs"], classes_splits["targets"]))[0]
        X_train, X_val, y_train, y_val = classes_splits["ecgs"][train_split], classes_splits["ecgs"][val_split], \
                                         classes_splits["targets"][train_split], classes_splits["targets"][val_split]
        return X_train, X_val, y_train, y_val, classes_splits["names"][train_split], classes_splits["names"][val_split]