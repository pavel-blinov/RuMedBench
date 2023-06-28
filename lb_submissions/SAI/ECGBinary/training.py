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
    logger = logging.getLogger('binary_baseline_training')
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p', filemode='w')
    fh = logging.FileHandler(Path(args.model_path) / "log_binary.txt")
    fh.setFormatter(logging.Formatter(log_format))
    logger.addHandler(fh)
    # Data preparing
    with open(Path(args.data_path)  / "train/idx2pathology.jsonl", "r") as f:
        classes = json.load(f)
    for class_name in classes:
        logger.info("---------- Working with %s ----------" % (classes[class_name]))
        X_train, X_val, y_train, y_val, names_train, names_val = get_dataset_baseline(args.data_path, classes[class_name], int(class_name), "train", args.random_state)
        X_public, names_public = get_dataset_baseline(args.data_path, classes[class_name], int(class_name), "test", args.random_state)
        model = CNN1dMultihead()
        opt = optim.AdamW(model.parameters(), lr=3e-3)

        train_ds = ECGRuDataset(X_train, y_train, names_train)
        val_ds = ECGRuDataset(X_val, y_val, names_val)
        test_public = ECGRuDataset(X_public, None, names_public)

        trainer = CNN1dTrainer(class_name = class_name, 
                            model = model, optimizer = opt, loss = nn.BCEWithLogitsLoss(),
                            train_dataset = train_ds, val_dataset = val_ds, test_dataset = test_public, 
                            model_path = args.model_path,
                            cuda_id = args.cuda_id)
        logger.info("---------- Model training started! ----------")
        trainer.train(args.num_epochs)
        with open(Path(args.model_path) / ( "models/" + class_name + "/ECG2Pathology.jsonl"), "r") as f:
            pred_i = json.load(f)
        if int(class_name) == 0:
            preds_dict = {k:[v] for k,v in pred_i.items()}
        else:
            for k in preds_dict:
                preds_dict[k].append(pred_i[k])

    out_fname = Path(args.model_path) / "ECG2Pathology.jsonl"
    with open(out_fname, 'w') as fw:
        for rec in preds_dict:
            json.dump({"record_name":rec, "labels":np.array(preds_dict[rec]).nonzero()[0].tolist()}, fw, ensure_ascii=False)
            fw.write('\n')

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Baselines training script (1d-CNN)')
    parser.add_argument('data_path', help='dataset path (path to the folder containing test and train subfolders)', type=str)
    parser.add_argument('model_path', help='path to save the model and logs', type=str)
    parser.add_argument('--cuda_id', help='CUDA device number on a single GPU; use -1 if yu want to work on CPU', type=int, default=1)
    parser.add_argument('--k', help='number of positive examples for class', type=int, default=11)
    parser.add_argument('--num_epochs', help='number of epochs', type=int, default=5)
    parser.add_argument('--random_state', help='random state number', type=int, default=19)
    args = parser.parse_args()
    Path(args.model_path).mkdir(parents = False, exist_ok = True)
    main(args)