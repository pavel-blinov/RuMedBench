import numpy as np
import pandas as pd
import random

import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, average_precision_score
from .utils import find_threshold_f1

import os
from tqdm import tqdm
import pickle

# from .preparing import create_prepared_data
import json


# from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"random seed set as {seed}")


##### ECG Dataset #####
class ECGRuDataset(Dataset):
    """ECG.RU Dataset."""

    def __init__(self, ecgs, labels, names):
        """
        Args:
            labels: array with labels
            ecgs: array with num_ch-lead ecgs
        """
        self.ecgs = ecgs
        if labels is not None:
             self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = None
        self.names = names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):        
        if self.labels is not None:
            sample = {'value': self.ecgs[idx], 'target': self.labels[idx], 'names': self.names[idx]}
        else:
            sample = {'value': self.ecgs[idx], 'names': self.names[idx]}
        return sample   
    

##### Multihead 1d-CNN model for the 1-st and 2-nd baselines #####
class CNN1dMultihead(nn.Module):
    def __init__(self, k=1, num_ch=12):
        super().__init__()
        """
        Args:
            num_ch: number of channels of an ecg-signal
            k: number of classes
        """
        self.layer1 = nn.Sequential(
            nn.Conv1d(num_ch, 24, 10, stride=2),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Conv1d(24, 48, 10, stride=2),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(6, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(48, 64, 10, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 10, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(10)
        )
        self.classification_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(128*10, 120),
            nn.ReLU(),
            nn.Linear(120, 160),
            nn.ReLU(),
            nn.Linear(160, 1)
        ) for i in range(k)])

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)
        preds = torch.stack([torch.squeeze(classification_layer(x)) for classification_layer in self.classification_layers])
        return torch.swapaxes(preds, 0, 1)
    
    
##### Trainer for 1d-CNN model #####
class CNN1dTrainer:
    """
    class_name - dict if multilabel (id2label), str in binary
    """
    def __init__(self, class_name, 
                 model, optimizer, loss,
                 train_dataset, val_dataset, test_dataset, model_path,
                 batch_size=128, cuda_id=1):
        
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_public = test_dataset
        
        self.result_output = {}

        self.batch_size = batch_size

        self.device = torch.device("cuda:" + str(cuda_id) if (torch.cuda.is_available() or cuda_id != -1) else "cpu")
        self.model = self.model.to(self.device)

        self.global_step = 0
        self.alpha = 0.8
        
        self.class_name = class_name
        
        self.result_output['class'] = class_name
        
        os.makedirs(model_path + "/models" + "/" +self.class_name, exist_ok=True)
        os.makedirs(model_path + "/summary" + "/" + self.class_name, exist_ok=True)
        os.makedirs(model_path + "/models" + "/" + self.class_name, exist_ok=True)
        self.writer = SummaryWriter(model_path + "/summary" + "/" + self.class_name)    
        self.model_path = model_path 

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)

    def train(self, num_epochs):
        
        model = self.model
        optimizer = self.optimizer
        
        self.train_loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True, batch_size=self.batch_size, num_workers=4)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, pin_memory=True, batch_size=len(self.val_dataset), num_workers=4)
        
        best_val = -38
        for epoch in tqdm(range(num_epochs)):
            model.train()
            train_logits = []
            train_gts = []

            for batch in self.train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items() if k != 'names'}
                optimizer.zero_grad()
                logits = model(batch['value']).squeeze()
                train_logits.append(logits.cpu().detach())
                train_gts.append(batch['target'].cpu())
                loss = self.loss(logits, batch['target'])
                loss.backward()
                optimizer.step()
                self.writer.add_scalar("Train Loss", loss.item(), global_step=self.global_step)
                self.global_step += 1

            train_logits = np.concatenate(train_logits)
            train_gts = np.concatenate(train_gts)

            if self.class_name != "multihead":
                train_logits = train_logits[:,None]
                train_gts = train_gts[:,None]

            res_ap = []
            for i in range(train_logits.shape[1]):
                res_ap.append(average_precision_score(train_gts[:,i], train_logits[:,i]))
            self.writer.add_scalar("Train AP/{}".format(self.class_name), np.mean(res_ap), global_step=epoch)

            model.eval()
            with torch.no_grad():
                for batch in self.val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items() if k != 'names'}
                    logits = model(batch['value']).cpu().squeeze()
                    gts = batch['target'].cpu()

                if self.class_name != "multihead":
                    logits = logits[:,None]
                    gts = gts[:,None]

                res_ap = []
                for i in range(logits.shape[1]):
                    res_ap.append(average_precision_score(gts[:,i], logits[:,i]))
                mean_val = np.mean(res_ap)

                if mean_val > best_val:
                    self.save_checkpoint(self.model_path + "/models" + "/" +self.class_name+"/best_checkpoint.pth")
                    best_val = mean_val
                    self.result_output['threshold_f1'] = find_threshold_f1(gts, logits)
                    self.test(self.model, self.test_public, "public", epoch)
                self.writer.add_scalar("Val AP/{}".format(self.class_name), mean_val, global_step=epoch)
        with open(self.model_path + "/models" + "/" +self.class_name+"/log.pickle", 'wb') as handle:
            pickle.dump(self.result_output, handle, protocol=pickle.HIGHEST_PROTOCOL)

       
    def test(self, model, test_dataset, name, epoch):
        model.eval()
        
        test_loader = DataLoader(test_dataset, shuffle=True, pin_memory=True, batch_size=len(test_dataset), num_workers=4)
        for batch in test_loader:
            names = batch['names']
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'names'}
            with torch.no_grad():
                logits = model(batch['value']).cpu().detach().squeeze()
            
        if self.class_name != "multihead":
            logits = logits[:,None]

        preds = []
        for i in range(logits.shape[1]):
            preds.append((logits[:,i] > self.result_output['threshold_f1'][i])*1)

        out_fname = self.model_path + "/models" + "/" +self.class_name + "/ECG2Pathology.jsonl"
        with open(out_fname, 'w') as fw:
            for rec in preds:
                res = dict(zip(names, rec.tolist()))
                json.dump(res, fw, ensure_ascii=False)
                fw.write('\n')