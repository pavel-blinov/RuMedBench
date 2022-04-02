# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
from sklearn.metrics import accuracy_score
from seqeval.metrics import f1_score
from seqeval.metrics import accuracy_score as seq_accuracy_score

def hit_at_3(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    hit_count = 0
    for l, row in zip(y_true, y_pred):
        hit_count += l in row
    return hit_count/float(len(y_true))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',
                        default='out/',
                        type=str,
                        help='The output directory with task results.')
    args = parser.parse_args()
    
    out_dir = args.out_dir
    if not os.path.exists(out_dir):
        raise ValueError('{} directory does not exist'.format(out_dir))

    files = set( os.listdir(out_dir) )

    metrics = {}
    label_id = 'code'
    for task in ['RuMedTop3', 'RuMedSymptomRec']:
        fname = '{}.jsonl'.format(task)
        if fname in files:
            fname = os.path.join(out_dir, fname)
            with open(fname) as f:
                result = [json.loads(line) for line in list(f)]
            gt = [d[label_id] for d in result]
            top1 = [d['prediction'][0] for d in result]
            top3 = [set(d['prediction']) for d in result]
            acc = accuracy_score(gt, top1)*100
            hit = hit_at_3(gt, top3)*100
            metrics[(task, 'acc')] = acc
            metrics[(task, 'hit3')] = hit
        else:
            print ('skip task {}'.format(task))
    
    for task, label_id in [('RuMedDaNet', 'answer'), ('RuMedNLI', 'gold_label')]:
        fname = '{}.jsonl'.format(task)
        if fname in files:
            fname = os.path.join(out_dir, fname)
            with open(fname) as f:
                result = [json.loads(line) for line in list(f)]
            gt = [d[label_id] for d in result]
            prediction = [d['prediction'] for d in result]
            acc = accuracy_score(gt, prediction)*100
            metrics[(task, 'acc')] = acc
        else:
            print ('skip task {}'.format(task))
    
    task = 'RuMedNER'
    fname = '{}.jsonl'.format(task)
    if fname in files:
        fname = os.path.join(out_dir, fname)
        with open(fname) as f:
            result = [json.loads(line) for line in list(f)]
        gt = [d['ner_tags'] for d in result]
        prediction = [d['prediction'] for d in result]
        for seq0, seq1 in zip(gt, prediction):
            assert len(seq0)==len(seq1)
        metrics[(task, 'acc')] = seq_accuracy_score(gt, prediction)*100
        metrics[(task, 'f1')] = f1_score(gt, prediction)*100
    else:
        print ('skip task {}'.format(task))

    top3_acc, top3_hit = metrics.get( ('RuMedTop3', 'acc'), 0 ), metrics.get( ('RuMedTop3', 'hit3'), 0 )
    rec_acc, rec_hit = metrics.get( ('RuMedSymptomRec', 'acc'), 0 ), metrics.get( ('RuMedSymptomRec', 'hit3'), 0 )
    danet_acc, nli_acc = metrics.get( ('RuMedDaNet', 'acc'), 0 ), metrics.get( ('RuMedNLI', 'acc'), 0 )
    ner_acc, ner_f1 = metrics.get( ('RuMedNER', 'acc'), 0 ), metrics.get( ('RuMedNER', 'f1'), 0 )

    overall = np.mean([
        (top3_acc+top3_hit)/2,
        (rec_acc+rec_hit)/2,
        danet_acc,
        nli_acc,
        (ner_acc+ner_f1)/2,
    ])

    result_line = '| {}\t| {:.2f} / {:.2f}\t|  {:.2f} / {:.2f}\t|   {:.2f}\t|  {:.2f}\t| {:.2f} / {:.2f}\t|  {:.2f}\t|'.format(
        out_dir,
        top3_acc, top3_hit,
        rec_acc, rec_hit,
        danet_acc,
        nli_acc,
        ner_acc, ner_f1,
        overall
    )
    print ('| Model\t\t| RuMedTop3\t| RuMedSymptomRec\t| RuMedDaNet\t| RuMedNLI\t| RuMedNER\t| Overall\t|')
    print (result_line)
