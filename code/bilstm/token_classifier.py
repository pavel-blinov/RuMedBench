# -*- coding: utf-8 -*-
import torch
from torch import nn
from torch.optim import AdamW
from torchtext import data
from torchtext.data import Field, BucketIterator

import os
import click
import json
import random
import numpy as np
import pandas as pd

from seqeval.metrics import f1_score, accuracy_score

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
SEED = 101
seed_everything(SEED)

class SequenceTaggingDataset(data.Dataset):
    @staticmethod
    def sort_key(example):
        for attr in dir(example):
            if not callable(getattr(example, attr)) and not attr.startswith('__'):
                return len(getattr(example, attr))
        return 0

    def __init__(self, list_of_lists, fields, **kwargs):
        examples = []
        columns = []
        for tup in list_of_lists:
            columns = list(tup)
            examples.append(data.Example.fromlist(columns, fields))

        super(SequenceTaggingDataset, self).__init__(examples, fields, **kwargs)

class Corpus(object):
    def __init__(self, input_folder, min_word_freq, batch_size):
        # list all the fields
        self.word_field = Field(lower=True)
        self.tag_field = Field(unk_token=None)
        
        parts = ['train', 'dev']
        p2data = {}
        for p in parts:
            fname = os.path.join(input_folder, '{}_v1.jsonl'.format(p))
            paired_lists = []
            with open(fname) as f:
                for line in f:
                    data = json.loads(line)
                    paired_lists.append( (data['tokens'], data['ner_tags']) )
            p2data[p] = paired_lists
        
        field_values = (('word', self.word_field), ('tag', self.tag_field))
        
        self.train_dataset = SequenceTaggingDataset( p2data['train'], fields=field_values )
        self.dev_dataset = SequenceTaggingDataset( p2data['dev'], fields=field_values )
        
        # convert fields to vocabulary list
        self.word_field.build_vocab(self.train_dataset.word, min_freq=min_word_freq)
        self.tag_field.build_vocab(self.train_dataset.tag)
        # create iterator for batch input
        self.train_iter, self.dev_iter = BucketIterator.splits(
            datasets=(self.train_dataset, self.dev_dataset),
            batch_size=batch_size
        )
        # prepare padding index to be ignored during model training/evaluation
        self.word_pad_idx = self.word_field.vocab.stoi[self.word_field.pad_token]
        self.tag_pad_idx = self.tag_field.vocab.stoi[self.tag_field.pad_token]

class BiLSTM(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, lstm_layers,
                 emb_dropout, lstm_dropout, fc_dropout, word_pad_idx):
        super().__init__()
        self.embedding_dim = embedding_dim
        # LAYER 1: Embedding
        self.embedding = nn.Embedding(
            num_embeddings=input_dim, 
            embedding_dim=embedding_dim, 
            padding_idx=word_pad_idx
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        # LAYER 2: BiLSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        # LAYER 3: Fully-connected
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # times 2 for bidirectional

    def forward(self, sentence):
        # sentence = [sentence length, batch size]
        # embedding_out = [sentence length, batch size, embedding dim]
        embedding_out = self.emb_dropout(self.embedding(sentence))
        # lstm_out = [sentence length, batch size, hidden dim * 2]
        lstm_out, _ = self.lstm(embedding_out)
        # ner_out = [sentence length, batch size, output dim]
        ner_out = self.fc(self.fc_dropout(lstm_out))
        return ner_out

    def init_weights(self):
        # to initialize all parameters from normal distribution
        # helps with converging during training
        for name, param in self.named_parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings(self, word_pad_idx):
        # initialize embedding for padding as zero
        self.embedding.weight.data[word_pad_idx] = torch.zeros(self.embedding_dim)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class NER(object):
    def __init__(self, model, data, optimizer_cls, loss_fn_cls, device=torch.device('cpu')):
        self.device = device
        self.model = model
        self.data = data
        self.optimizer = optimizer_cls(model.parameters(), lr=0.0015, weight_decay=0.01)
        self.loss_fn = loss_fn_cls(ignore_index=self.data.tag_pad_idx)
        self.loss_fn = self.loss_fn.to(self.device)

    def accuracy(self, preds, y):
        max_preds = preds.argmax(dim=1, keepdim=True)  # get the index of the max probability
        non_pad_elements = (y != self.data.tag_pad_idx).nonzero()  # prepare masking for paddings
        correct = max_preds[non_pad_elements].squeeze(1).eq(y[non_pad_elements])
        denom = torch.cuda.FloatTensor([y[non_pad_elements].shape[0]])
        return correct.sum() / denom

    def epoch(self):
        epoch_loss = 0
        epoch_acc = 0
        self.model.train()
        for batch in self.data.train_iter:
            # text = [sent len, batch size]
            text = batch.word.to(self.device)
            # tags = [sent len, batch size]
            true_tags = batch.tag.to(self.device)
            self.optimizer.zero_grad()
            pred_tags = self.model(text)
            # to calculate the loss and accuracy, we flatten both prediction and true tags
            # flatten pred_tags to [sent len, batch size, output dim]
            pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
            # flatten true_tags to [sent len * batch size]
            true_tags = true_tags.view(-1)
            batch_loss = self.loss_fn(pred_tags, true_tags)
            batch_acc = self.accuracy(pred_tags, true_tags)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
            epoch_acc += batch_acc.item()
        return epoch_loss / len(self.data.train_iter), epoch_acc / len(self.data.train_iter)

    def evaluate(self, iterator):
        epoch_loss = 0
        epoch_acc = 0
        self.model.eval()
        cum = 0
        whole_gt_seq, whole_pred_seq = [], []
        with torch.no_grad():
            # similar to epoch() but model is in evaluation mode and no backprop
            for batch in iterator:
                text = batch.word.to(self.device)
                true_tags = batch.tag.to(self.device)
                pred_tags = self.model(text)
                
                #[sentence length, batch size, output dim]
                for i, (row, tag_row) in enumerate(zip(text.T, true_tags.T)):
                    mask = row!=1
                    gt_seq = [self.data.tag_field.vocab.itos[j.item()] for j in tag_row[mask]]
                    pred_idx = pred_tags[:,i,:].argmax(-1)[mask]
                    pred_seq = [self.data.tag_field.vocab.itos[j.item()] for j in pred_idx]
                    whole_gt_seq.append(gt_seq)
                    whole_pred_seq.append(pred_seq)                
                pred_tags = pred_tags.view(-1, pred_tags.shape[-1])
                
                true_tags = true_tags.view(-1)
                batch_loss = self.loss_fn(pred_tags, true_tags)
                batch_acc = self.accuracy(pred_tags, true_tags)
                epoch_loss += batch_loss.item()
                epoch_acc += batch_acc.item()
        acc = accuracy_score(whole_gt_seq, whole_pred_seq)
        f1 = f1_score(whole_gt_seq, whole_pred_seq)
        return epoch_loss / len(iterator), acc, f1, whole_gt_seq, whole_pred_seq

    def train(self, n_epochs):
        for epoch in range(n_epochs):
            train_loss, train_acc = self.epoch()
            dev_loss, dev_acc, dev_f1, _, _ = self.evaluate(self.data.dev_iter)
            print (f'Epoch {epoch:02d}\t| Dev Loss: {dev_loss:.3f} | Dev Acc: {dev_acc * 100:.2f}% | Dev F1: {dev_f1 * 100:.2f}%')
    
    def infer(self, tokens):
        tokens = [t.lower() for t in tokens]
        self.model.eval()
        # transform to indices based on corpus vocab
        numericalized_tokens = [self.data.word_field.vocab.stoi[t] for t in tokens]
        # begin prediction
        token_tensor = torch.LongTensor(numericalized_tokens)
        token_tensor = token_tensor.unsqueeze(-1)
        predictions = self.model(token_tensor.to(self.device))
        # convert results to tags
        top_predictions = predictions.argmax(-1)
        predicted_tags = [self.data.tag_field.vocab.itos[t.item()] for t in top_predictions]
        return predicted_tags

@click.command()
@click.option('--task-name',
    default='RuMedNER',
    type=click.Choice(['RuMedNER']),
    help='The name of the task to run.')
@click.option('--device',
    default=-1,
    help='Gpu to train the model on.')
def main(task_name, device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{task_name} task')

    base_path = os.path.abspath( os.path.join(os.path.dirname( __file__ ) ) )
    out_dir = os.path.join(base_path, 'out')

    base_path = os.path.abspath( os.path.join(base_path, '../..') )

    data_path = os.path.join(base_path, 'data', task_name)

    corpus = Corpus(
        input_folder=data_path,
        min_word_freq=1,
        batch_size=32
    )
    print (f'Train set: {len(corpus.train_dataset)} sentences')
    print (f'Dev set: {len(corpus.dev_dataset)} sentences')

    bilstm = BiLSTM(
        input_dim=len(corpus.word_field.vocab),
        embedding_dim=300,
        hidden_dim=256,
        output_dim=len(corpus.tag_field.vocab),
        lstm_layers=2,
        emb_dropout=0.5,
        lstm_dropout=0.1,
        fc_dropout=0.25,
        word_pad_idx=corpus.word_pad_idx
    )

    bilstm.init_weights()
    bilstm.init_embeddings(word_pad_idx=corpus.word_pad_idx)
    print (f'The model has {bilstm.count_parameters():,} trainable parameters.')
    print (bilstm)

    ner = NER(
        model=bilstm.to(device),
        data=corpus,
        optimizer_cls=AdamW,
        loss_fn_cls=nn.CrossEntropyLoss,
        device=device
    )

    ner.train(20)

    test_data = pd.read_json(os.path.join(data_path, 'test_v1.jsonl'), lines=True)

    out_fname = os.path.join(out_dir, task_name+'.jsonl')
    with open(out_fname, 'w') as fw:
        for i, true, tokens in zip(test_data.idx, test_data.ner_tags, test_data.tokens):
            prediction = ner.infer(tokens)
            rec = {'idx': i, 'ner_tags': true, 'prediction': prediction}
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')

if __name__ == '__main__':
    main()
