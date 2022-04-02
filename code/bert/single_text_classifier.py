# -*- coding: utf-8 -*-
import gc
import os
import pandas as pd
import numpy as np
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig
from transformers.optimization import AdamW

import argparse
from scipy.special import expit
from keras.preprocessing.sequence import pad_sequences

from utils import seed_everything, seed_worker

def encode_texts(tokenizer, sentences):
    bs = 20000
    input_ids, attention_masks = [], []
    for _, i in enumerate(range(0, len(sentences), bs)):
        b_sentences = ['[CLS] ' + sentence + ' [SEP]' for sentence in sentences[i:i+bs]]
        tokenized_texts = [tokenizer.tokenize(sent) for sent in b_sentences]
        b_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
        b_input_ids = pad_sequences(b_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')
        b_attention_masks = []
        for seq in b_input_ids:
            seq_mask = [float(i>0) for i in seq]
            b_attention_masks.append(seq_mask)

        attention_masks.append(b_attention_masks)
        input_ids.append(b_input_ids)
    input_ids, attention_masks = np.vstack(input_ids), np.vstack(attention_masks)
    return input_ids, attention_masks

def hit_at_n(y_true, y_pred, index2label, n=3):
    assert len(y_true) == len(y_pred)
    hit_count = 0
    for l, row in zip(y_true, y_pred):
        order = (np.argsort(row)[::-1])[:n]
        order = [index2label[i] for i in order]
        order = set(order)
        hit_count += int(l in order)
    return hit_count/float(len(y_true))

SEED = 128
seed_everything(SEED)

MAX_LEN = 256

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu',
                        default=None,
                        type=int,
                        required=True,
                        help='The index of the gpu to run.')
    parser.add_argument('--task_name',
                        default='',
                        type=str,
                        required=True,
                        help='The name of the task to run.')
    parser.add_argument('--bert_type',
                        default='',
                        type=str,
                        required=True,
                        help='The type of BERT model (bert or pool).')
    return parser

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bert_type=='pool': #get model type of BERT model
        from utils import PoolBertForSequenceClassification as BertModel
    else:
        from transformers import BertForSequenceClassification as BertModel

    task_name = args.task_name

    base_path = os.path.abspath( os.path.join(os.path.dirname( __file__ ) ) )
    out_dir = os.path.join(base_path, 'out')
    model_path = os.path.join(base_path, 'models/rubert_cased_L-12_H-768_A-12_pt/')

    base_path = os.path.abspath( os.path.join(base_path, '../..') )

    parts = ['train', 'dev', 'test']
    data_path = os.path.join(base_path, 'data', task_name)

    text1_id, label_id, index_id = 'symptoms', 'code', 'idx'
    if task_name=='RuMedTop3':
        pass
    elif task_name=='RuMedSymptomRec':
        pass
    else:
        raise ValueError('unknown task')
    
    part2indices = {p:set() for p in parts}
    all_ids, sentences, labels = [], [], []
    for p in parts:
        fname = '{}_v1.jsonl'.format(p)
        with open(os.path.join( data_path, fname)) as f:
            for line in f:
                data = json.loads(line)
                s1 = data[text1_id]
                sentences.append( s1 )
                labels.append( data[label_id] )
                idx = data[index_id]
                all_ids.append( idx )
                part2indices[p].add( idx )
    all_ids = np.array(all_ids)
    print ('len(total)', len(sentences))

    code_set = set(labels)
    l2i = {code:i for i, code in enumerate(sorted(code_set))}
    i2l = {l2i[l]:l for l in l2i}
    print ( 'len(l2i)', len(l2i) )

    tokenizer = BertTokenizer.from_pretrained(
        os.path.join(base_path, model_path),
        do_lower_case=True,
        max_length=MAX_LEN
    )

    input_ids, attention_masks = encode_texts(tokenizer, sentences)

    label_indices = np.array([l2i[l] for l in labels])

    labels = np.zeros((input_ids.shape[0], len(l2i)))
    for _, i in enumerate(label_indices):
        labels[_, i] = 1
    
    # prepare test data loader
    test_ids = part2indices['test']
    test_mask = np.array([sid in test_ids for sid in all_ids])
    test_ids = all_ids[test_mask]
    tst_inputs, tst_masks, tst_labels = input_ids[test_mask], attention_masks[test_mask], labels[test_mask]

    tst_inputs = torch.tensor(tst_inputs)
    tst_masks = torch.tensor(tst_masks)
    tst_labels = torch.tensor(tst_labels)

    test_data = TensorDataset(tst_inputs, tst_masks, tst_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8, worker_init_fn=seed_worker)

    batch_size = 4
    epochs = 25
    lr = 3e-5
    max_grad_norm = 1.0

    cv_res = {}
    for fold in range(1):
        best_dev_score = -1
        seed_everything(SEED)
        train_ids = part2indices['train']
        dev_ids = part2indices['dev']

        train_mask = np.array([sid in train_ids for sid in all_ids])
        dev_mask = np.array([sid in dev_ids for sid in all_ids])
        
        input_ids_train, attention_masks_train, labels_train = input_ids[train_mask], attention_masks[train_mask], labels[train_mask]
        input_ids_dev, attention_masks_dev, labels_dev = input_ids[dev_mask], attention_masks[dev_mask], labels[dev_mask]
        print ('fold', fold, input_ids_train.shape, input_ids_dev.shape)

        input_ids_train = torch.tensor(input_ids_train)
        attention_masks_train = torch.tensor(attention_masks_train)
        labels_train = torch.tensor(labels_train)

        train_data = TensorDataset(input_ids_train, attention_masks_train, labels_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        ##prediction_dataloader
        input_ids_dev = torch.tensor(input_ids_dev)
        attention_masks_dev = torch.tensor(attention_masks_dev)
        labels_dev = torch.tensor(labels_dev)
        prediction_data = TensorDataset(input_ids_dev, attention_masks_dev, labels_dev)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        ## take appropriate config and init a BERT model
        config_path = os.path.join( base_path, model_path, 'bert_config.json' )
        conf = BertConfig.from_json_file( config_path )
        conf.num_labels = len(l2i)
        model = BertModel(conf)
        output_model_file = os.path.join( base_path, model_path, 'pytorch_model.bin' )
        model.load_state_dict(torch.load(output_model_file), strict=False)
        model = model.cuda()

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]

        # This variable contains all of the hyperparemeter information our training loop needs
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr, correct_bias=False)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs)

        train_loss = []
        for _ in range(epochs):
            model.train(); torch.cuda.empty_cache()
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                optimizer.zero_grad()

                outputs = model( b_input_ids, attention_mask=b_input_mask, labels=b_labels )
                loss, logits = outputs[:2]
                train_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1
            avg_train_loss = tr_loss/nb_tr_steps

            ### val
            model.eval()
            predictions = []
            tr_loss, nb_tr_steps = 0, 0
            for step, batch in enumerate(prediction_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model( b_input_ids, attention_mask=b_input_mask, labels=b_labels )
                    loss, logits = outputs[:2]
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)
            predictions = expit(np.vstack(predictions))
            edev_loss = tr_loss/nb_tr_steps
            
            y_indices = np.argmax(labels_dev.detach().cpu().numpy(), axis=1)
            dev_codes = [i2l[i] for i in y_indices]
            
            dev_acc = hit_at_n(dev_codes, predictions, i2l, n=1)*100
            dev_hit_at3 = hit_at_n(dev_codes, predictions, i2l, n=3)*100
            print ('{} epoch {} average train_loss: {:.6f}\tdev_loss: {:.6f}\tdev_acc {:.2f}\tdev_hit_at3 {:.2f}'.format(task_name, _, avg_train_loss, edev_loss, dev_acc, dev_hit_at3))

            score = (dev_acc+dev_hit_at3)/2
            if score>best_dev_score: # compute result for test part and store to out file, if we found better model
                best_dev_score = score
                cv_res[fold] = (dev_acc, dev_hit_at3)
        
                predictions, true_labels = [], []
                for batch in test_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    logits = outputs[1].detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    predictions.append(logits)
                    true_labels.append(label_ids)
                predictions = expit(np.vstack(predictions))
                true_labels = np.concatenate(true_labels)
                assert len(true_labels) == len(predictions)
                recs = []
                for idx, l, row in zip(test_ids, true_labels, predictions):
                    gt = i2l[np.argmax(l)]
                    order = (np.argsort(row)[::-1])[:3]
                    pred = [i2l[i] for i in order]
                    recs.append( (idx, gt, pred) )
                
                out_fname = os.path.join(out_dir, task_name+'.jsonl')
                with open(out_fname, 'w') as fw:
                    for rec in recs:
                        data = {index_id:rec[0], label_id:rec[1], 'prediction':rec[2]}
                        json.dump(data, fw, ensure_ascii=False)
                        fw.write('\n')
        del model; gc.collect(); torch.cuda.empty_cache()

    dev_acc, dev_hit_at3 = cv_res[0]
    print ('\ntask scores {}: {:.2f}/{:.2f}'.format(task_name, dev_acc, dev_hit_at3))
