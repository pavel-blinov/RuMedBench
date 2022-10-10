# -*- coding: utf-8 -*-
import gc
import os
import pandas as pd
import numpy as np
import json
import click
import pathlib

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig
from transformers.optimization import AdamW

from scipy.special import expit
from keras.preprocessing.sequence import pad_sequences
from sklearn import metrics

from utils import seed_everything, seed_worker

def encode_text_pairs(tokenizer, sentences):
    bs = 20000
    input_ids, attention_masks, token_type_ids = [], [], []
    
    text1_max = int(MAX_LEN*.75) #leave 75% of token lens to premise text
    for _, i in enumerate(range(0, len(sentences), bs)):
        tokenized_texts = []
        for sentence in sentences[i:i+bs]:
            p1 = ['[CLS]']+tokenizer.tokenize( sentence[0] )
            p2 = ['[SEP]']+tokenizer.tokenize( sentence[1] )+['[SEP]']
            text2_max = MAX_LEN-len(p1[:text1_max])
            final_tokens = p1[:text1_max]+p2[:text2_max]
            arr = np.array(final_tokens)
            mask = arr == '[SEP]'
            tokenized_texts.append(final_tokens)

        b_input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

        b_input_ids = pad_sequences(b_input_ids, maxlen=MAX_LEN, dtype='long', truncating='post', padding='post')

        b_token_type_ids = []
        for i, row in enumerate(b_input_ids):
            row = np.array(row)
            mask = row==tokenizer.convert_tokens_to_ids('[SEP]')
            idx = np.where(mask)[0][0]
            idx1 = np.where(mask)[0][1]
            token_type_row = np.zeros(row.shape[0], dtype=np.int)
            token_type_row[idx+1:idx1+1] = 1
            b_token_type_ids.append(token_type_row)

        b_attention_masks = []
        for seq in b_input_ids:
            seq_mask = [float(i>0) for i in seq]
            b_attention_masks.append(seq_mask)

        attention_masks.append(b_attention_masks)
        input_ids.append(b_input_ids)
        token_type_ids.append(b_token_type_ids)
    input_ids, attention_masks = np.vstack(input_ids), np.vstack(attention_masks)
    token_type_ids = np.vstack(token_type_ids)

    return input_ids, attention_masks, token_type_ids
SEED = 128
seed_everything(SEED)

MAX_LEN = 512

@click.command()
@click.option('--task-name',
                default='RuMedNLI',
                type=click.Choice(['RuMedDaNet', 'RuMedNLI']),
                help='The name of the task to run.')
@click.option('--device',
                default=-1,
                help='Gpu to train the model on.')
@click.option('--data-path',
                default='../../../MedBench_data/',
                help='Path to the data files.')
@click.option('--bert-type',
                default='bert',
                help='BERT model variant.')
def main(task_name, data_path, device, bert_type):
    print(f'\n{task_name} task')

    if bert_type=='pool': #get model type of BERT model
        from utils import PoolBertForSequenceClassification as BertModel
    else:
        from transformers import BertForSequenceClassification as BertModel

    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

    out_dir = pathlib.Path('.').absolute()
    data_path = pathlib.Path(data_path).absolute() / task_name

    parts = ['train', 'dev', 'test']

    if task_name=='RuMedNLI':
        l2i = {'neutral': 0, 'entailment':1, 'contradiction': 2}
        text1_id, text2_id, label_id, index_id = 'ru_sentence1', 'ru_sentence2', 'gold_label', 'pairID'
    elif task_name=='RuMedDaNet':
        l2i = {'нет': 0, 'да':1}
        text1_id, text2_id, label_id, index_id = 'context', 'question', 'answer', 'pairID'        
    else:
        raise ValueError('unknown task')
    dummy_label = list(l2i.keys())[0]
    
    part2indices = {p:set() for p in parts}
    all_ids, sentences, labels = [], [], []
    for p in parts:
        fname = '{}.jsonl'.format(p)
        with open(os.path.join( data_path, fname)) as f:
            for line in f:
                data = json.loads(line)
                s1, s2 = data[text1_id], data[text2_id]
                sentences.append( (s1, s2) )
                labels.append( data.get(label_id, dummy_label) )
                idx = data[index_id]
                all_ids.append( idx )
                part2indices[p].add( idx )
    all_ids = np.array(all_ids)
    print ('len(total)', len(sentences))

    i2l = {l2i[l]:l for l in l2i}
    print ( 'len(l2i)', len(l2i), l2i )

    tokenizer = BertTokenizer.from_pretrained(
        out_dir / 'models/rubert_cased_L-12_H-768_A-12_pt/',
        do_lower_case=True,
        max_length=MAX_LEN
    )

    input_ids, attention_masks, token_type_ids = encode_text_pairs(tokenizer, sentences)

    label_indices = np.array([l2i[l] for l in labels])

    labels = np.zeros((input_ids.shape[0], len(l2i)))
    for _, i in enumerate(label_indices):
        labels[_, i] = 1
    
    # prepare test data loader
    test_ids = part2indices['test']
    test_mask = np.array([sid in test_ids for sid in all_ids])
    test_ids = all_ids[test_mask]
    tst_inputs, tst_masks, tst_labels = input_ids[test_mask], attention_masks[test_mask], labels[test_mask]
    tst_type_ids_dev = token_type_ids[test_mask]

    tst_inputs = torch.tensor(tst_inputs)
    tst_masks = torch.tensor(tst_masks)
    tst_labels = torch.tensor(tst_labels)
    tst_type_ids_dev = torch.tensor(tst_type_ids_dev)

    test_data = TensorDataset(tst_inputs, tst_masks, tst_type_ids_dev, tst_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8, worker_init_fn=seed_worker)

    batch_size = 16
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
        token_type_ids_train = token_type_ids[train_mask]
        input_ids_dev, attention_masks_dev, labels_dev = input_ids[dev_mask], attention_masks[dev_mask], labels[dev_mask]
        token_type_ids_dev = token_type_ids[dev_mask]
        print ('fold', fold, input_ids_train.shape, input_ids_dev.shape)

        input_ids_train = torch.tensor(input_ids_train)
        attention_masks_train = torch.tensor(attention_masks_train)
        labels_train = torch.tensor(labels_train)
        token_type_ids_train = torch.tensor(token_type_ids_train)

        train_data = TensorDataset(input_ids_train, attention_masks_train, token_type_ids_train, labels_train)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        ##prediction_dataloader
        input_ids_dev = torch.tensor(input_ids_dev)
        attention_masks_dev = torch.tensor(attention_masks_dev)
        labels_dev = torch.tensor(labels_dev)
        token_type_ids_dev = torch.tensor(token_type_ids_dev)

        prediction_data = TensorDataset(input_ids_dev, attention_masks_dev, token_type_ids_dev, labels_dev)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        ## take appropriate config and init a BERT model
        config_path = out_dir / 'models/rubert_cased_L-12_H-768_A-12_pt/bert_config.json'
        conf = BertConfig.from_json_file( config_path )
        conf.num_labels = len(l2i)
        model = BertModel(conf)
        ## preload it with weights
        output_model_file = out_dir / 'models/rubert_cased_L-12_H-768_A-12_pt/pytorch_model.bin'
        model.load_state_dict(torch.load(output_model_file), strict=False)
        model = model.to(device)

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
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
                optimizer.zero_grad()

                outputs = model( b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels )
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
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
                with torch.no_grad():
                    outputs = model( b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels )
                    loss, logits = outputs[:2]
                    tr_loss += loss.item()
                    nb_tr_steps += 1
                logits = logits.detach().cpu().numpy()
                predictions.append(logits)
            predictions = expit(np.vstack(predictions))
            edev_loss = tr_loss/nb_tr_steps

            y_indices, pred = np.argmax(labels_dev, axis=1), np.argmax(predictions, axis=1)
            dev_acc = metrics.accuracy_score(y_indices, pred)*100
            print ('{} epoch {} average train_loss: {:.6f}\tdev_loss: {:.6f}\tdev_acc {:.2f}'.format(task_name, _, avg_train_loss, edev_loss, dev_acc))
            
            if dev_acc>best_dev_score: # compute result for test part and store to out file, if we found better model
                best_dev_score = dev_acc
                cv_res[fold] = (best_dev_score)

                predictions, true_labels = [], []
                for batch in test_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_token_type_ids, b_labels = batch
                    with torch.no_grad():
                        outputs = model( b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels )
                    
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
                    pred = i2l[np.argmax(row)]
                    recs.append( (idx, gt, pred) )
                
                out_fname = out_dir / f'{task_name}.jsonl'
                with open(out_fname, 'w') as fw:
                    for rec in recs:
                        data = {index_id:rec[0], label_id:rec[2]}
                        json.dump(data, fw, ensure_ascii=False)
                        fw.write('\n')
        del model; gc.collect(); torch.cuda.empty_cache()

    dev_acc = cv_res[0]
    print ('\ntask scores {}: {:.2f}'.format(task_name, dev_acc))


if __name__ == '__main__':
    main()
