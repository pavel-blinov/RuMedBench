import json
import pathlib

import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def hit_at_n(y_true, y_pred, n=3):
    assert len(y_true) == len(y_pred)
    hit_count = 0
    for l, row in zip(y_true, y_pred):
        order = (np.argsort(row)[::-1])[:n]
        hit_count += int(l in order)
    return round(hit_count / float(len(y_true)) * 100, 2)


def encode_text(tfidf, text_data, labels, l2i, mode='train'):
    if mode == 'train':
        X = tfidf.fit_transform(text_data)
    else:
        X = tfidf.transform(text_data)
    y = labels.map(l2i)
    return X, y


def logits2codes(logits, i2l, n=3):
    codes = []
    for row in logits:
        order = np.argsort(row)[::-1]
        codes.append([i2l[i] for i in order[:n]])
    return codes


@click.command()
@click.option('--task-name',
                default='RuMedTop3',
                type=click.Choice(['RuMedTop3', 'RuMedSymptomRec']),
                help='The name of the task to run.')
def main(task_name):
    print(f'\n{task_name} task')

    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    out_path = base_path / 'code' / 'linear_models' / 'out'
    data_path = base_path / 'data' / task_name
    
    train_data = pd.read_json(data_path / 'train_v1.jsonl', lines=True)
    dev_data = pd.read_json(data_path / 'dev_v1.jsonl', lines=True)
    test_data = pd.read_json(data_path / 'test_v1.jsonl', lines=True)

    text_id = 'symptoms'
    label_id = 'code' 
    index_id = 'idx'

    i2l = dict(enumerate(sorted(train_data[label_id].unique())))
    l2i = {label: i for i, label in i2l.items()}

    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 8))
    clf = LogisticRegression(penalty='l2', C=10, multi_class='ovr', n_jobs=10, verbose=1)

    X, y = encode_text(tfidf, train_data[text_id], train_data[label_id], l2i)

    clf.fit(X, y)

    X_val, y_val = encode_text(tfidf, dev_data[text_id], dev_data[label_id], l2i, mode='val')
    y_val_pred = clf.predict_proba(X_val)

    accuracy = hit_at_n(y_val, y_val_pred, n=1)
    hit_3 = hit_at_n(y_val, y_val_pred, n=3)
    print (f'\n{task_name} task scores on dev set: {accuracy} / {hit_3}')

    X_test, _ = encode_text(tfidf, test_data[text_id], test_data[label_id], l2i, mode='test')
    y_test_pred = clf.predict_proba(X_test)

    test_codes = logits2codes(y_test_pred, i2l)

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
