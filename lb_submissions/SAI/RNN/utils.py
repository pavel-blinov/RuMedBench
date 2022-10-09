import os
from string import punctuation
import random

from nltk.tokenize import ToktokTokenizer
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from typing import List, Dict, Union, Tuple, Set, Any

TOKENIZER = ToktokTokenizer()


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


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def preprocess(text, tokenizer=TOKENIZER):
    res = []
    tokens = tokenizer.tokenize(text.lower())
    for t in tokens:
        if t not in punctuation:
            res.append(t.strip(punctuation))
    return res


class DataPreprocessor(Dataset):
    
    def __init__(self, x_data, y_data, word2index, label2index, 
                 sequence_length=128, pad_token='PAD', unk_token='UNK', preprocessing=True):
        
        super().__init__()
        
        self.x_data = []
        self.y_data = len(x_data)*[list(label2index.values())[0]]
        if type(y_data)!=type(None):
            self.y_data = y_data.map(label2index)
        
        self.word2index = word2index
        self.sequence_length = sequence_length
        
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_index = self.word2index[self.pad_token]

        self.preprocessing = preprocessing
        
        self.load(x_data)

    def load(self, data):
        
        for text in data:
            if self.preprocessing:
                words = preprocess(text)
            else:
                words = text
            indexed_words = self.indexing(words)
            self.x_data.append(indexed_words)
    
    def indexing(self, tokenized_text):
        unk_index = self.word2index[self.unk_token]
        return [self.word2index.get(token, unk_index) for token in tokenized_text]
    
    def padding(self, sequence):
        sequence = sequence + [self.pad_index] * (max(self.sequence_length - len(sequence), 0))
        return sequence[:self.sequence_length]
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = self.x_data[idx]
        x = self.padding(x)
        x = torch.Tensor(x).long()
        
        if type(self.y_data)==type(None):
            y = None
        else:
            y = self.y_data[idx]
        
        return x, y


def preprocess_for_tokens(
        tokens: List[str]
    ) -> List[str]:

    return tokens

class DataPreprocessorNer(Dataset):
    
    def __init__(
            self, 
            x_data: pd.Series, 
            y_data: pd.Series, 
            word2index: Dict[str, int], 
            label2index: Dict[str, int], 
            sequence_length: int = 128, 
            pad_token: str = 'PAD',
            unk_token: str = 'UNK'
        ) -> None:
        
        super().__init__()

        self.word2index = word2index
        self.label2index = label2index
        
        self.sequence_length = sequence_length
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_index = self.word2index[self.pad_token]
        self.unk_index = self.word2index[self.unk_token]

        self.x_data = self.load(x_data, self.word2index)
        self.y_data = self.load(y_data, self.label2index)

    
    def load(
            self, 
            data: pd.Series, 
            mapping: Dict[str, int]
        ) -> List[List[int]]:
        
        indexed_data = []
        for case in data:
            processed_case = preprocess_for_tokens(case)
            indexed_case = self.indexing(processed_case, mapping)
            indexed_data.append(indexed_case)

        return indexed_data
    

    def indexing(
            self, 
            tokenized_case: List[str], 
            mapping: Dict[str, int]
        ) -> List[int]:

        return [mapping.get(token, self.unk_index) for token in tokenized_case]
    

    def padding(
            self, 
            sequence: List[int]
        ) -> List[int]:
        sequence = sequence + [self.pad_index] * (max(self.sequence_length - len(sequence), 0))
        return sequence[:self.sequence_length]
    

    def __len__(self):
        return len(self.x_data)
    

    def __getitem__(
            self, 
            idx: int
        ) -> Tuple[torch.tensor, torch.tensor]:

        x = self.x_data[idx]
        y = self.y_data[idx]

        assert len(x) > 0

        x = self.padding(x)
        y = self.padding(y)

        x = torch.tensor(x, dtype=torch.int64)
        y = torch.tensor(y, dtype=torch.int64)
        
        return x, y
