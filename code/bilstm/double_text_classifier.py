from collections import defaultdict
import json
import pathlib

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
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


def preprocess_two_seqs(text1, text2, seq_len):
    seq1_len = int(seq_len * 0.75)
    seq2_len = seq_len - seq1_len
    
    tokens1 = preprocess(text1)[:seq1_len]
    tokens2 = preprocess(text2)[:seq2_len]

    return tokens1 + tokens2


def build_vocab(text_data, min_freq=1):
    word2freq = defaultdict(int)
    word2index = {'PAD': 0, 'UNK': 1}

    for text in text_data:
        for token in text:
            word2freq[token] += 1

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
            
            test_preds.append(torch.argmax(pred, dim=1).cpu().numpy())
            test_true.append(y.cpu().numpy())
            
        pbar.update(x.shape[0])
    pbar.close()
    
    test_preds = np.concatenate(test_preds)
    
    if mode == 'dev':
        test_true = np.concatenate(test_true)
        mean_test_loss = np.mean(test_losses)
        accuracy = round(accuracy_score(test_true, test_preds) * 100, 2)
        return mean_test_loss, accuracy

    else:
        return test_preds


def train(train_data, dev_data, model, optimizer, criterion, device, n_epochs=50, max_patience=3):
    
    losses = []
    best_accuracy = 0.
    
    patience = 0
    best_test_loss = 10.
    
    for epoch in range(n_epochs):
        
        losses = train_step(train_data, model, optimizer, criterion, device, losses, epoch)
        mean_dev_loss, accuracy = eval_step(dev_data, model, criterion, device)

        if accuracy > best_accuracy:
            best_accuracy = accuracy

        print(f'\nDev loss: {mean_dev_loss} \naccuracy: {accuracy}')
        
        if mean_dev_loss < best_test_loss:
            best_test_loss = mean_dev_loss
        elif patience == max_patience:
            print(f'Dev loss did not improve in {patience} epochs, early stopping')
            break
        else:
            patience += 1
    return best_accuracy


@click.command()
@click.option('--task-name',
                default='RuMedNLI',
                type=click.Choice(['RuMedDaNet', 'RuMedNLI']),
                help='The name of the task to run.')
@click.option('--device',
                default=-1,
                help='Gpu to train the model on.')
@click.option('--seq-len',
                default=256,
                help='Max sequence length.')
def main(task_name, device, seq_len):
    print(f'\n{task_name} task')

    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    out_path = base_path / 'code' / 'bilstm' / 'out'
    data_path = base_path / 'data' / task_name
    
    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    index_id = 'pairID'
    if task_name == 'RuMedNLI':
        l2i = {'neutral': 0, 'entailment': 1, 'contradiction': 2}
        text1_id = 'ru_sentence1'
        text2_id = 'ru_sentence2'
        label_id = 'gold_label' 

    elif task_name == 'RuMedDaNet':
        l2i = {'нет': 0, 'да': 1}
        text1_id = 'context'
        text2_id = 'question'
        label_id = 'answer'
    else:
        raise ValueError('unknown task')

    i2l = {i: label for label, i in l2i.items()}

    text_data_train = [preprocess_two_seqs(text1, text2, seq_len) for text1, text2 in \
        zip(train_data[text1_id], train_data[text2_id])]
    text_data_dev = [preprocess_two_seqs(text1, text2, seq_len) for text1, text2 in \
        zip(dev_data[text1_id], dev_data[text2_id])]
    text_data_test = [preprocess_two_seqs(text1, text2, seq_len) for text1, text2 in \
        zip(test_data[text1_id], test_data[text2_id])]

    word2index = build_vocab(text_data_train, min_freq=0)
    print(f'Total: {len(word2index)} tokens')

    train_dataset = DataPreprocessor(text_data_train, train_data[label_id], word2index, l2i, \
        sequence_length=seq_len, preprocessing=False)
    dev_dataset = DataPreprocessor(text_data_dev, dev_data[label_id], word2index, l2i, \
        sequence_length=seq_len, preprocessing=False)
    test_dataset = DataPreprocessor(text_data_test, test_data[label_id], word2index, l2i, \
        sequence_length=seq_len, preprocessing=False)

    gen = torch.Generator()
    gen.manual_seed(SEED)
    train_dataset = DataLoader(train_dataset, batch_size=64, worker_init_fn=seed_worker,  generator=gen)
    dev_dataset = DataLoader(dev_dataset, batch_size=64, worker_init_fn=seed_worker, generator=gen)
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
    
    accuracy = train(train_dataset, dev_dataset, model, optimizer, criterion, device)
    print (f'\n{task_name} task score on dev set: {accuracy}')

    test_preds = eval_step(test_dataset, model, criterion, device, mode='test')

    recs = []
    for i, true, pred in zip(test_data[index_id], test_data[label_id], test_preds):
        recs.append({index_id: i, label_id: true, 'prediction': i2l[pred]})

    out_fname = out_path / f'{task_name}.jsonl'
    with open(out_fname, 'w') as fw:
        for rec in recs:
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    main()
