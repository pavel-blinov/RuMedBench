# -*- coding: utf-8 -*-
import gc
import os
import ast
import json
import pandas as pd
import numpy as np
from sklearn import model_selection
from collections import Counter

SEED = 53

def df2jsonl(in_df, fname, code2freq, th=10):
    with open(fname, 'w', encoding='utf-8') as fw:
        for idx, symptoms, code in zip(in_df.new_event_id, in_df.symptoms, in_df.code):
            if code in code2freq and code2freq[code]>th:
                data = {
                    'idx':idx,
                    'symptoms': symptoms,
                    'code': code,
                }
                json.dump(data, fw, ensure_ascii=False)
                fw.write("\n")

def ner2jsonl(in_df, ids, fname):
    trim_ids = np.array([s.split('_')[0] for s in in_df['Sentence#'].values])

    with open(fname, 'w') as fw:
        for i in ids:
            mask = trim_ids==i
            sample_ids = np.array(list(set(in_df['Sentence#'].values[mask])))
            order = np.argsort([int(k.split('_')[-1]) for k in sample_ids])
            sample_ids = sample_ids[order]
            for idx in sample_ids:
                sub_mask = in_df['Sentence#'].values==idx
                tokens = list(in_df.Word[sub_mask].values)
                ner_tags = list(in_df.Tag[sub_mask].values)
                assert len(tokens)==len(ner_tags)
                data = {
                    'idx':idx,
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                }
                json.dump(data, fw, ensure_ascii=False)
                fw.write("\n")

def jsonl2jsonl(source, target):
    with open(target, 'w') as fw:
        with open(source) as f:
            for line in f:
                data = json.loads(line)
                selected = {field:data[field] for field in ['ru_sentence1', 'ru_sentence2', 'gold_label', 'pairID']}
                json.dump(selected, fw, ensure_ascii=False)
                fw.write("\n")

if __name__ == '__main__':
    base_path = os.path.abspath( os.path.join(os.path.dirname( __file__ ), '..') )

    data_path = os.path.join(base_path, 'data/')

    data_fname = os.path.join(data_path, 'raw', 'RuMedPrimeData.tsv')

    if not os.path.isfile(data_fname):
        raise ValueError('Have you downloaded the data file RuMedPrimeData.tsv and place it into data/ directory?')

    base_split_names = ['train', 'dev', 'test']
    ## prepare data for RuMedTop3 task
    df = pd.read_csv(data_fname, sep='\t')
    df['code'] = df.icd10.apply(lambda s: s.split('.')[0])
    # parts'll be list of [train, dev, test]
    parts = np.split(df.sample(frac=1, random_state=SEED), [int(0.735*len(df)), int(0.8675*len(df))])

    code2freq = dict(parts[0]['code'].value_counts())

    for i, part in enumerate(base_split_names):
        df2jsonl(
            parts[i],
            os.path.join(data_path, 'RuMedTop3', '{}_v1.jsonl'.format(part)),
            code2freq
        )

    ## prepare data for RuMedSymptomRec task
    df.drop(columns=['code'], inplace=True)
    rec_markup = pd.read_csv( os.path.join(data_path, 'raw', 'rec_markup.csv') )
    df = pd.merge(df, rec_markup, on='new_event_id')

    mask = ~df.code.isna().values
    df = df.iloc[mask]

    symptoms_reduced = []
    for text, span in zip(df.symptoms, df.keep_spans):
        span = ast.literal_eval(span)
        reduced_text = (''.join([text[s[0]:s[1]] for s in span])).strip()
        symptoms_reduced.append(reduced_text)
    df['symptoms'] = symptoms_reduced

    parts = np.split(df.sample(frac=1, random_state=SEED), [int(0.735*len(df)), int(0.8675*len(df))])

    code2freq = dict(parts[0]['code'].value_counts())

    for i, part in enumerate(base_split_names):
        df2jsonl(
            parts[i],
            os.path.join(data_path, 'RuMedSymptomRec', '{}_v1.jsonl'.format(part)),
            code2freq
        )

    ## prepare data for RuMedNER task
    df = pd.read_csv( os.path.join(data_path, 'raw', 'RuDReC.csv') )

    d = Counter(df['Sentence#'].apply(lambda s: s.split('_')[0]))
    ids = np.array(list(d.keys()))
    lens = np.array(list(d.values()))
    lens = np.array([len(str(i)) for i in lens])

    sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=75, random_state=7)
    for fold, (train_idx, test_idx) in enumerate(sss.split(ids, lens)):
        train_ids, test_ids = ids[train_idx], ids[test_idx]
    
    sss = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=75, random_state=6)
    for fold, (train_idx, test_idx) in enumerate(sss.split(train_ids, lens[train_idx])):
        train_ids, dev_ids = train_ids[train_idx], train_ids[test_idx]
    parts = [train_ids, dev_ids, test_ids]

    for i, part in enumerate(base_split_names):
        ner2jsonl(
            df,
            parts[i],
            os.path.join(data_path, 'RuMedNER', '{}_v1.jsonl'.format(part))
        )

    ## prepare data for RuMedNLI task
    for part in base_split_names:
        fname = os.path.join(data_path, 'raw', 'ru_mli_{}_v1.jsonl'.format(part))
        jsonl2jsonl(
            fname,
            os.path.join(data_path, 'RuMedNLI', '{}_v1.jsonl'.format(part))
        )
