import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


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