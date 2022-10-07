import json
import pathlib

import click
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

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
        l2i = {'1': 1, '2': 2, '3': 3, '4': 4}
        question_id = 'question'
        label_id = 'answer'
    else:
        raise ValueError('unknown task')

    i2l = {i: label for label, i in l2i.items()}

    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 8))

    text_data = test_data[question_id]

    X = tfidf.fit_transform(text_data)

    sims = []
    for l in sorted(list(l2i.keys())):
        option_X = tfidf.transform( test_data[l] )
        sim = cosine_similarity(X, option_X).diagonal()
        sims.append(sim)
    sims = np.array(sims).T

    recs = []
    for i, pred in zip(test_data[index_id], sims):
        recs.append({index_id: i, label_id: i2l[1+np.argmax(pred)]})

    out_fname = out_path / f'{task_name}.jsonl'
    with open(out_fname, 'w') as fw:
        for rec in recs:
            json.dump(rec, fw, ensure_ascii=False)
            fw.write('\n')


if __name__ == '__main__':
    main()
