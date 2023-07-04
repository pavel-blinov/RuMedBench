import torch.optim as optim
import torch.nn as nn
import numpy as np

from ECGBaselineLib.datasets import get_dataset_baseline
from ECGBaselineLib.neurobaseline import set_seed, ECGRuDataset, CNN1dTrainer, CNN1dMultihead

import sys
import logging

import argparse

from pathlib import Path
import json


def main(args):
    # Fix seed
    set_seed(seed = args.random_state)
    # Logger
    logger = logging.getLogger('baseline_multihead_training')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode='w')
    fh = logging.FileHandler(Path(args.model_path) / "log_multihead.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    # Data preparing
    with open(Path(args.data_path)  / "train/idx2pathology.jsonl", "r") as f:
        classes = json.load(f)
    logger.info("---------- Working with multihead model ----------")
    X_train, X_val, y_train, y_val, names_train, names_val = get_dataset_baseline(args.data_path, "train", args.random_state)
    X_public, names_public = get_dataset_baseline(args.data_path, "test", args.random_state)
    model = CNN1dMultihead(k=73)
    opt = optim.AdamW(model.parameters(), lr=3e-3)

    train_ds = ECGRuDataset(X_train, y_train, names_train)
    val_ds = ECGRuDataset(X_val, y_val, names_val)
    test_public = ECGRuDataset(X_public, None, names_public)

    trainer = CNN1dTrainer(class_name = "multihead", 
                        model = model, optimizer = opt, loss = nn.BCEWithLogitsLoss(),
                        train_dataset = train_ds, val_dataset = val_ds, test_dataset = test_public, 
                        model_path = args.model_path,
                        cuda_id = args.cuda_id)
    logger.info("---------- Model training started! ----------")
    trainer.train(args.num_epochs)

    out_fname = Path(args.model_path) / "ECG2Pathology.jsonl"
    with open(Path(args.model_path) / ( "models/" + "multihead" + "/ECG2Pathology.jsonl"), 'r') as fw:
        for i, line in enumerate(fw):
            line = json.loads(line)
            if i == 0:
                preds_dict = {k:[v] for k,v in line.items()}
            else:
                for k in preds_dict:
                    preds_dict[k].append(line[k])

    out_fname = Path(args.model_path) / "ECG2Pathology.jsonl"
    with open(out_fname, 'w') as fw:
        for rec in preds_dict:
            json.dump({"record_name":rec, "labels":np.array(preds_dict[rec]).nonzero()[0].tolist()}, fw, ensure_ascii=False)
            fw.write('\n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Baselines training script (1d-CNN)')
    parser.add_argument('data_path', help='dataset path (path to the folder containing test and train subfolders)', type=str)
    parser.add_argument('model_path', help='path to save the model and logs', type=str)
    parser.add_argument('--cuda_id', help='CUDA device number on a single GPU; use -1 if you want to work on CPU', type=int, default=0)
    parser.add_argument('--k', help='number of positive examples for class', type=int, default=11)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=5)
    parser.add_argument('--random_state', help='random state number', type=int, default=19)
    args = parser.parse_args()
    main(args)