from collections import defaultdict
import json
import pathlib

import click
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import preprocess, seed_everything, seed_worker, DataPreprocessor

SEED = 101
seed_everything(SEED)
class Classifier(nn.Module):
    
    def __init__(self, n_classes, vocab_size, emb_dim=300, hidden_dim=256):
        
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
    
        self.embedding_layer = nn.Embedding(vocab_size, self.emb_dim)
        self.lstm_layer = nn.LSTM(self.emb_dim, self.hidden_dim, batch_first=True, num_layers=2,
                                  bidirectional=True)
        self.linear_layer = nn.Linear(self.hidden_dim * 2, n_classes)
        
    def forward(self, x):
        x = self.embedding_layer(x)
        _, (hidden, _) = self.lstm_layer(x)
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], axis=1)
        return self.linear_layer(hidden)


def hit_at_n(y_true, y_pred, n=3):
    assert len(y_true) == len(y_pred)
    hit_count = 0
    for l, row in zip(y_true, y_pred):
        order = (np.argsort(row)[::-1])[:n]
        hit_count += int(l in order)
    return round(hit_count / float(len(y_true)) * 100, 2)


def logits2codes(logits, i2l, n=3):
    codes = []
    for row in logits:
        order = np.argsort(row)[::-1]
        codes.append([i2l[i] for i in order[:n]])
    return codes


def build_vocab(text_data, min_freq=1):
    word2freq = defaultdict(int)
    word2index = {'PAD': 0, 'UNK': 1}

    for text in text_data:
        for t in preprocess(text):
            word2freq[t] += 1

    for word, freq in word2freq.items():
        if freq > min_freq:
            word2index[word] = len(word2index)
    return word2index


def train_step(data, model, optimizer, criterion, device, losses, epoch):
    
    model.train()
    
    pbar = tqdm(total=len(data.dataset), desc=f'Epoch: {epoch + 1}')
    
    for x, y in data:
        
        x = x.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        
        loss = criterion(pred, y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        pbar.set_postfix(train_loss = np.mean(losses[-100:]))
        pbar.update(x.shape[0])
        
    pbar.close()
        
    return losses

def eval_step(data, model, criterion, device, mode='dev'):
    
    test_losses = []
    test_preds = []
    test_true = []
    
    pbar = tqdm(total=len(data.dataset), desc=f'Predictions on {mode} set')
    
    model.eval()
    
    for x, y in data:
        
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            
            pred = model(x)
            
            loss = criterion(pred, y)
            test_losses.append(loss.item())
            
            test_preds.append(pred.cpu().numpy())
            test_true.append(y.cpu().numpy())
            
        pbar.update(x.shape[0])
    pbar.close()
    
    test_preds = np.concatenate(test_preds)
    
    if mode == 'dev':
        test_true = np.concatenate(test_true)
        mean_test_loss = np.mean(test_losses)
        accuracy = hit_at_n(test_true, test_preds, n=1)
        hit_3 = hit_at_n(test_true, test_preds, n=3)
        return mean_test_loss, accuracy, hit_3

    else:
        return test_preds


def train(train_data, dev_data, model, optimizer, criterion, device, n_epochs=50, max_patience=3):
    
    losses = []
    best_metrics = [0.0, 0.0]
    
    patience = 0
    best_test_loss = 10.
    
    for epoch in range(n_epochs):
        
        losses = train_step(train_data, model, optimizer, criterion, device, losses, epoch)
        mean_dev_loss, accuracy, hit_3 = eval_step(dev_data, model, criterion, device)

        if accuracy > best_metrics[0] and hit_3 > best_metrics[1]:
            best_metrics = [accuracy, hit_3]

        print(f'\nDev loss: {mean_dev_loss} \naccuracy: {accuracy}, hit@3: {hit_3}')
        
        if mean_dev_loss < best_test_loss:
            best_test_loss = mean_dev_loss
        elif patience == max_patience:
            print(f'Dev loss did not improve in {patience} epochs, early stopping')
            break
        else:
            patience += 1
    return best_metrics


@click.command()
@click.option('--task-name',
                default='RuMedTop3',
                type=click.Choice(['RuMedTop3', 'RuMedSymptomRec']),
                help='The name of the task to run.')
@click.option('--device',
                default=-1,
                help='Gpu to train the model on.')
def main(task_name, device):
    print(f'\n{task_name} task')

    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    out_path = base_path / 'code' / 'bilstm' / 'out'
    data_path = base_path / 'data' / task_name
    
    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    text_id = 'symptoms'
    label_id = 'code' 
    index_id = 'idx'

    i2l = dict(enumerate(sorted(train_data[label_id].unique())))
    l2i = {label: i for i, label in i2l.items()}

    word2index = build_vocab(train_data[text_id], min_freq=0)
    print(f'Total: {len(word2index)} tokens')

    train_dataset = DataPreprocessor(train_data[text_id], train_data[label_id], word2index, l2i)
    dev_dataset = DataPreprocessor(dev_data[text_id], dev_data[label_id], word2index, l2i)
    test_dataset = DataPreprocessor(test_data[text_id], test_data[label_id], word2index, l2i)

    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_dataset = DataLoader(train_dataset, batch_size=64, worker_init_fn=seed_worker,  generator=gen)
    dev_dataset = DataLoader(dev_dataset, batch_size=64, worker_init_fn=seed_worker,  generator=gen)
    test_dataset = DataLoader(test_dataset, batch_size=64, worker_init_fn=seed_worker,  generator=gen)

    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    model = Classifier(n_classes=len(l2i), vocab_size=len(word2index))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters())

    model = model.to(device)
    criterion = criterion.to(device)
    
    accuracy, hit_3 = train(train_dataset, dev_dataset, model, optimizer, criterion, device)
    print (f'\n{task_name} task scores on dev set: {accuracy} / {hit_3}')

    test_logits = eval_step(test_dataset, model, criterion, device, mode='test')
    test_codes = logits2codes(test_logits, i2l)

    recs = []
    for i, true, pred in zip(test_data[index_id], test_data[label_id], test_codes):
        recs.append({index_id: i, label_id: true, 'prediction': pred})

    out_fname = out_path / f'{task_name}.jsonl'
    with open(out_fname, 'w') as fw:
        for rec in recs:
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    main()
