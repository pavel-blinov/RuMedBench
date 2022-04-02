# -*- coding: utf-8 -*-
import gc
import os
import json
import numpy as np
import random
import click
from seqeval.metrics import accuracy_score, f1_score
import sklearn_crfsuite

SEED = 128
random.seed(SEED)
np.random.seed(SEED)

def load_sents(fname):
    sents = []
    with open(fname) as f:
        for line in f:
            data = json.loads(line)
            idx = data['idx']
            codes = data['ner_tags']
            tokens = data['tokens']
            sample = []
            for token, code in zip(tokens,codes):
                sample.append( (token, code) )
            sents.append( (idx, sample) )
    return sents

def word2features(sent, i):
    word = sent[i][0]
    
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

@click.command()
@click.option('--task-name',
    default='RuMedNER',
    type=click.Choice(['RuMedNER']),
    help='The name of the task to run.'
)

def main(task_name):
    print(f'\n{task_name} task')
    base_path = os.path.abspath( os.path.join(os.path.dirname( __file__ ) ) )
    out_dir = os.path.join(base_path, 'out')

    base_path = os.path.abspath( os.path.join(base_path, '../..') )

    parts = ['train', 'dev', 'test']
    data_path = os.path.join(base_path, 'data', task_name)

    text1_id, label_id, index_id = 'tokens', 'ner_tags', 'idx'
    part2data = {}
    for p in parts:
        fname = os.path.join( data_path, '{}_v1.jsonl'.format(p) )
        sents = load_sents(fname)
        part2data[p] = sents

    part2feat = {}
    for p in parts:
        p_X = [sent2features(s) for idx, s in part2data[p]]
        p_y = [sent2labels(s) for idx, s in part2data[p]]
        p_ids = [idx for idx, _ in part2data[p]]
        part2feat[p] = (p_X, p_y, p_ids)

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.01,
        max_iterations=200,
        all_possible_transitions=True,
        verbose=True
    )
    X_train, y_train = part2feat['train'][0], part2feat['train'][1]
    crf = crf.fit(X_train, y_train)

    X_dev = part2feat['dev'][0]
    y_pred_dev = crf.predict(X_dev)

    y_dev = part2feat['dev'][1]
    dev_acc, dev_f1 = accuracy_score(y_dev, y_pred_dev)*100, f1_score(y_dev, y_pred_dev)*100

    print ('\n{} task scores on dev set: {:.2f}/{:.2f}'.format(task_name, dev_acc, dev_f1))

    X_test = part2feat['test'][0]
    y_pred_test = crf.predict(X_test)
    out_fname = os.path.join(out_dir, task_name+'.jsonl')
    with open(out_fname, 'w') as fw:
        for idx, labels, prediction in zip(part2feat['test'][-1], part2feat['test'][1], y_pred_test):
            data = {index_id:idx, label_id:labels, 'prediction':prediction}
            json.dump(data, fw, ensure_ascii=False)
            fw.write('\n')

if __name__ == '__main__':
    main()
