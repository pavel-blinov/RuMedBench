import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split



def prepare_data(X, y):
    df = pd.DataFrame(X).reset_index(drop=True)
    df.loc[:, 'noise'] = df[['baseline_drift', 'static_noise', 'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker']].isna().sum(axis=1).apply(lambda x: 1 if x==6 else 0)
    df = df[['age', 'sex', 'validated_by_human', 'site', 'device', 'noise']]
    if y is None:
        return df
    df['targets'] = pd.DataFrame(y)
    return df


##### Split for the AutoML baseline ######
def get_dataset_baseline(data_path, class_id, dtype, random_state):
    assert dtype in ["train", "test"]
    classes_splits = {"ecgs":[], "targets":[], "names": []}
    metadata = pd.read_json(Path(data_path) / (dtype + "/" + dtype + ".jsonl"), lines=True)
    for _, signal in metadata.iterrows():
        if dtype == "train":
            classes_splits["ecgs"].append(signal[2:-6])
            classes_splits["targets"].append((class_id in signal["labels"]) * 1)
        else:
            classes_splits["ecgs"].append(signal[2:-5])
            classes_splits["names"].append(signal["record_name"])
    classes_splits["targets"] = np.array(classes_splits["targets"])
    if dtype == "test":
        del classes_splits["targets"]
        return prepare_data(classes_splits['ecgs'], None), classes_splits["names"]
    else:
        del classes_splits["names"]
        X_train, X_val, y_train, y_val = train_test_split(
                                                        classes_splits["ecgs"], 
                                                        classes_splits["targets"], 
                                                        test_size=0.2, 
                                                        random_state=random_state,
                                                        stratify=classes_splits["targets"]
                                                        )
        return prepare_data(X_train, y_train), prepare_data(X_val, y_val)