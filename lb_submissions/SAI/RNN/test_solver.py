import json
import pathlib

import click
import numpy as np
import pandas as pd

import torch
import joblib
from utils import preprocess, DataPreprocessor
from double_text_classifier import Classifier
from sklearn.metrics.pairwise import cosine_similarity

seq_len = 256

@click.command()
@click.option('--task-name',
                default='RuMedTest',
                type=click.Choice(['RuMedTest']),
                help='The name of the task to run.')
@click.option('--data-path',
                default='../../../MedBench_data/',
                help='Path to the data files.')
def main(task_name, data_path):
    print(f'\n{task_name} task')

    out_path = pathlib.Path('.').absolute()
    data_path = pathlib.Path(data_path).absolute() / task_name
    
    test_data = pd.read_json(data_path / 'test.jsonl', lines=True)

    index_id = 'idx'
    if task_name == 'RuMedTest':
        options = ['1', '2', '3', '4']
        question_id = 'question'
        label_id = 'answer'
    else:
        raise ValueError('unknown task')

    word2index = joblib.load('word2index.pkl')
    l2i = joblib.load('l2i.pkl')

    model = Classifier(n_classes=len(l2i), vocab_size=len(word2index))
    model.load_state_dict(torch.load('model.bin'))
    model.eval();

    text_data_test = [preprocess(text1) for text1 in test_data['question']]

    test_dataset = DataPreprocessor(text_data_test, None, word2index, l2i, \
        sequence_length=seq_len, preprocessing=False)

    q_vecs = []
    for x, _ in test_dataset:
        with torch.no_grad():
            x = model.embedding_layer(x[None, :])
            _, (hidden, _) = model.lstm_layer(x)
            hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], axis=1).detach().cpu().numpy()
            q_vecs.append(hidden[0])
    q_vecs = np.array(q_vecs)

    sims = []
    for option in options:
        text_data_test = [preprocess(text1) for text1 in test_data[option]]
        test_dataset = DataPreprocessor(text_data_test, None, word2index, l2i, \
            sequence_length=seq_len, preprocessing=False)
        
        option_vecs = []
        for x, _ in test_dataset:
            with torch.no_grad():
                x = model.embedding_layer(x[None, :])
                _, (hidden, _) = model.lstm_layer(x)
                hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], axis=1).detach().cpu().numpy()
                option_vecs.append(hidden[0])
        option_vecs = np.array(option_vecs)

        sim = cosine_similarity(q_vecs, option_vecs).diagonal()
        sims.append(sim)
    sims = np.array(sims).T

    recs = []
    for i, pred in zip(test_data[index_id], sims):
        recs.append( { index_id: i, label_id: str(1+np.argmax(pred)) } )

    out_fname = out_path / f'{task_name}.jsonl'
    with open(out_fname, 'w') as fw:
        for rec in recs:
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    main()
