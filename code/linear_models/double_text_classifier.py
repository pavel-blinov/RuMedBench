import json
import pathlib

import click
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def preprocess_sentences(column1, column2):
    return [sent1 + ' ' + sent2 for sent1, sent2 in zip(column1, column2)]


def encode_text(tfidf, text_data, labels, l2i, mode='train'):
    if mode == 'train':
        X = tfidf.fit_transform(text_data)
    else:
        X = tfidf.transform(text_data)
    y = labels.map(l2i)
    return X, y


@click.command()
@click.option('--task-name',
                default='RuMedNLI',
                type=click.Choice(['RuMedDaNet', 'RuMedNLI']),
                help='The name of the task to run.')
def main(task_name):
    print(f'\n{task_name} task')

    base_path = pathlib.Path(__file__).absolute().parent.parent.parent
    out_path = base_path / 'code' / 'linear_models' / 'out'
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

    text_data_train = preprocess_sentences(train_data[text1_id], train_data[text2_id])
    text_data_dev = preprocess_sentences(dev_data[text1_id], dev_data[text2_id])
    text_data_test = preprocess_sentences(test_data[text1_id], test_data[text2_id])

    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 8))
    clf = LogisticRegression(penalty='l2', C=10, multi_class='ovr', n_jobs=10, verbose=1)

    X, y = encode_text(tfidf, text_data_train, train_data[label_id], l2i)

    clf.fit(X, y)

    X_val, y_val = encode_text(tfidf, text_data_dev, dev_data[label_id], l2i, mode='val')
    y_val_pred = clf.predict(X_val)
    accuracy = round(accuracy_score(y_val, y_val_pred) * 100, 2)
    print (f'\n{task_name} task score on dev set: {accuracy}')

    X_test, _ = encode_text(tfidf, text_data_test, test_data[label_id], l2i, mode='test')
    y_test_pred = clf.predict(X_test)

    recs = []
    for i, true, pred in zip(test_data[index_id], test_data[label_id], y_test_pred):
        recs.append({index_id: i, label_id: true, 'prediction': i2l[pred]})

    out_fname = out_path / f'{task_name}.jsonl'
    with open(out_fname, 'w') as fw:
        for rec in recs:
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    main()
