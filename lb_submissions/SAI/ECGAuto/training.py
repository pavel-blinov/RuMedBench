from ECGBaselineLib.autobaseline import lama_train
from ECGBaselineLib.datasets import get_dataset_baseline

import sys
import logging
import os
import argparse
from pathlib import Path
import json
import joblib
import numpy as np


def main(args):
    # Logger
    logger = logging.getLogger('automl_baseline_training')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(Path(args.model_path + "/summary/") / 'log_automl.txt')
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    # Получение необходимых классов
    with open(Path(args.data_path)  / "train/idx2pathology.jsonl", "r") as f:
        classes = json.load(f)
    for class_name in classes:
        os.makedirs(args.model_path + "/models/" + class_name, exist_ok=True)
        logger.info("---------- Working with LAMA and {} class ----------".format(class_name))
        X_train, X_val = get_dataset_baseline(args.data_path, int(class_name), "train", args.random_state)
        X_public, public_names = get_dataset_baseline(args.data_path, int(class_name), "test", args.random_state)
        # Модель и всё такое
        pub_res, model = lama_train([X_train, X_val, X_public], random_seed = args.random_state) * 1
        if class_name == '0':
            preds_dict = {key: [val] for key, val in dict(zip(public_names, pub_res)).items()}
        else:
            for i, key in enumerate(preds_dict):
                preds_dict[key].append(pub_res[i])
        joblib.dump(model, args.model_path + "/models/" + class_name + "/model.pkl")

    out_fname = Path(args.model_path) / "ECG2Pathology.jsonl"
    with open(out_fname, 'w') as fw:
        for rec in preds_dict:
            json.dump({"record_name":rec, "labels":np.array(preds_dict[rec]).nonzero()[0].tolist()}, fw, ensure_ascii=False)
            fw.write('\n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Baselines training script (LAMA)')
    parser.add_argument('data_path', help='dataset path (path to the folder containing test and train subfolders)', type=str)
    parser.add_argument('model_path', help='path to save the model and logs', type=str)
    parser.add_argument('--random_state', help='random state number', type=int, default=19)
    args = parser.parse_args()

    os.makedirs(args.model_path + "/models/", exist_ok=True)
    os.makedirs(args.model_path + "/summary/", exist_ok=True)
    main(args)