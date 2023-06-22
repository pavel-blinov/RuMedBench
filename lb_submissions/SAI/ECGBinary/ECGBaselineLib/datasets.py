import numpy as np
import pandas as pd
import os
from pathlib import Path
    
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from pathlib import Path

from tqdm import tqdm


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
    
    
##### Split for the N models baseline ######
def get_dataset_baseline(data_path, class_name, class_id, dtype, random_state):
    assert dtype in ["train", "test"]
    classes_splits = {"ecgs":[], "targets":[], "names":[]}
    metadata = pd.read_json(Path(data_path) / (dtype + "/" + dtype + ".jsonl"), lines=True)
    for signal in (Path(data_path) / dtype).glob("*.npy"):
        signal_name = signal.name[:signal.name.rfind('/')-3]
        classes_splits["names"].append(signal_name)
        with open(signal, "rb") as f:
            signal_value = np.load(f, allow_pickle=True)
        classes_splits['ecgs'].append(signal_value)
        if dtype == "train":
            classes_splits["targets"].append((class_id in metadata.loc[metadata.record_name == signal_name, "labels"].item()) * 1)
    classes_splits["targets"] = np.array(classes_splits["targets"])
    if dtype == "test":
        del classes_splits["targets"]
        return classes_splits['ecgs'], classes_splits["names"]
    else:
        X_train, X_val, y_train, y_val, names_train, names_val = train_test_split(
                                                            classes_splits["ecgs"], 
                                                            classes_splits["targets"], 
                                                            classes_splits["names"],
                                                            test_size=0.33, 
                                                            random_state=random_state,
                                                            stratify=classes_splits["targets"]
                                                            )
        return X_train, X_val, y_train, y_val, names_train, names_val