# -*- coding: utf-8 -*-
import gc
import os
import numpy as np
import json
import itertools

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertConfig
from transformers.optimization import AdamW

from seqeval.metrics import accuracy_score, f1_score

from keras.preprocessing.sequence import pad_sequences

from utils import seed_everything, seed_worker
from single_text_classifier import setup_parser

def tokenize_and_preserve_labels(tokenizer, sentence, text_labels):
    tokenized_sentence = []
    labels = []
    word_indices = []
    for i, (word, label) in enumerate(zip(sentence, text_labels)):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)
        word_indices.extend([i] * n_subwords)

    return tokenized_sentence, labels, word_indices

def colapse_token(accum_tok, accum_gt, accum_p):
    assert len(set(accum_gt))==1 # check consistency in ground truth labels
    orig_gt = accum_gt[0]
    orig_token = ''.join(accum_tok) # join sub-tokens
    orig_p = accum_p[0] # take leading token tag as a final tag
    #orig_p = Counter(accum_p).most_common()[0][0]
    return orig_token, orig_gt, orig_p

def untokenize(i2t, inp_ids, word_indices, pred, gt_labels):
    all_restored_pred, all_restored_labels = [], []
    for inp_row, indices, p, l in zip(inp_ids, word_indices, pred, gt_labels):
        restored_pred, restored_labels = [], []
        tokens = tokenizer.convert_ids_to_tokens(inp_row)
        
        accum_tok, accum_gt, accum_p = [], [], []
        prev = indices[0]
        for tok, uidx, p_i, l_i in zip(tokens, indices, p, l):
            if tok.startswith('##'):
                tok = tok[2:] # strip '##' in a sub-token, like '##ing' --> 'ing'
            if i2t[l_i] != 'PAD' or len(accum_tok)>0:
                p_tag, gt_tag = i2t[p_i], i2t[l_i]
                if prev!=uidx:
                    if len(accum_tok)>0:
                        t, gt, pt = colapse_token(accum_tok, accum_gt, accum_p)
                        
                        restored_labels.append(gt)
                        restored_pred.append(pt)

                    accum_tok, accum_gt, accum_p = [], [], []
                    prev = uidx
                
                if gt_tag!='PAD':
                    accum_tok.append(tok)
                    accum_gt.append(gt_tag)
                    accum_p.append(p_tag)

        if len(accum_tok)>0:
            t, gt, pt = colapse_token(accum_tok, accum_gt, accum_p)
            print ('call', accum_tok, accum_gt, accum_p)
            restored_labels.append(gt)
            restored_pred.append(pt)
        all_restored_labels.append(restored_labels)
        all_restored_pred.append(restored_pred)
    return all_restored_labels, all_restored_pred

SEED = 128
seed_everything(SEED)

MAX_LEN = 256

if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bert_type=='pool': #get model type of BERT model
        from utils import PoolBertForTokenClassification as BertModel
    else:
        from transformers import BertForTokenClassification as BertModel

    task_name = args.task_name

    base_path = os.path.abspath( os.path.join(os.path.dirname( __file__ ) ) )
    out_dir = os.path.join(base_path, 'out')
    model_path = os.path.join(base_path, 'models/rubert_cased_L-12_H-768_A-12_pt/')

    base_path = os.path.abspath( os.path.join(base_path, '../..') )

    parts = ['train', 'dev', 'test']
    data_path = os.path.join(base_path, 'data', task_name)

    text1_id, label_id, index_id = 'tokens', 'ner_tags', 'idx'
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

    tag_values = sorted(list(set( itertools.chain.from_iterable(labels) )))
    tag_values.append('PAD')
    tag2idx = {t: i for i, t in enumerate(tag_values)}
    idx2tag = {i: t for i, t in enumerate(tag_values)}
    print ('tag2idx', tag2idx)
    print ( 'len(tag2idx)', len(tag2idx) )

    tokenizer = BertTokenizer.from_pretrained(
        model_path,
        do_lower_case=True,
        max_length=MAX_LEN
    )

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(tokenizer, sent, labs)
        for sent, labs in zip(sentences, labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    word_indices = [token_label_pair[-1] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences(
        [tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
        maxlen=MAX_LEN,
        dtype='long',
        value=0.0,
        truncating='post',
        padding='post'
    )
    print ('input_ids.shape', input_ids.shape)

    tags = pad_sequences(
        [[tag2idx[l] for l in lab] for lab in labels],
        maxlen=MAX_LEN,
        dtype='long',
        value=tag2idx['PAD'],
        truncating='post',
        padding='post'
    )
    print ('tags.shape', tags.shape)

    word_indices = pad_sequences(
        [[l for l in lab] for lab in word_indices],
        maxlen=MAX_LEN,
        dtype='long',
        value=-1,
        truncating='post',
        padding='post'
    )

    attention_masks = np.array([[float(i != 0.0) for i in ii] for ii in input_ids])
    print ('attention_masks.shape', attention_masks.shape)

    # prepare test data loader
    test_ids = part2indices['test']
    test_mask = np.array([sid in test_ids for sid in all_ids])
    tst_inputs, tst_masks, tst_tags = input_ids[test_mask], attention_masks[test_mask], tags[test_mask]
    tst_word_indices = word_indices[test_mask]

    tst_inputs = torch.tensor(tst_inputs)
    tst_masks = torch.tensor(tst_masks)
    tst_tags = torch.tensor(tst_tags)

    test_data = TensorDataset(tst_inputs, tst_masks, tst_tags)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=8, worker_init_fn=seed_worker)

    batch_size = 32
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
        
        tr_inputs, tr_masks, tr_tags = input_ids[train_mask], attention_masks[train_mask], tags[train_mask]
        tr_word_indices = word_indices[train_mask]
        dev_inputs, dev_masks, dev_tags = input_ids[dev_mask], attention_masks[dev_mask], tags[dev_mask]
        dev_word_indices = word_indices[dev_mask]
        print ('fold', fold, tr_inputs.shape, dev_inputs.shape)

        tr_inputs = torch.tensor(tr_inputs)
        dev_inputs = torch.tensor(dev_inputs)

        tr_masks = torch.tensor(tr_masks)
        dev_masks = torch.tensor(dev_masks)

        tr_tags = torch.tensor(tr_tags)
        dev_tags = torch.tensor(dev_tags)

        train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        valid_data = TensorDataset(dev_inputs, dev_masks, dev_tags)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        config_path = os.path.join(model_path, 'bert_config.json')
        conf = BertConfig.from_json_file( config_path )
        conf.num_labels = len(tag2idx)
        model = BertModel(conf)
        output_model_file = os.path.join(model_path, 'pytorch_model.bin')
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

        for _ in range(epochs):
            model.train()
            total_loss = 0

            # Training loop
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                model.zero_grad()

                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs[0]
                
                loss.backward()
                total_loss += loss.item()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(train_dataloader)

            # After the completion of each training epoch, measure our performance on our validation set
            model.eval()
            edev_loss = 0
            predictions, true_labels = [], []
            for batch in valid_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                logits = outputs[1].detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                edev_loss += outputs[0].mean().item()
                predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                true_labels.extend(label_ids)

            edev_loss = edev_loss / len(valid_dataloader)

            restored_labels, restored_pred = untokenize(idx2tag, dev_inputs, dev_word_indices, predictions, true_labels)
            dev_acc = accuracy_score(restored_labels, restored_pred)*100
            dev_f1 = f1_score(restored_labels, restored_pred)*100
            
            print ('{} epoch {} average train_loss: {:.6f}\tdev_loss: {:.6f}\tdev_acc {:.2f}\tdev_f1 {:.2f}'.format(task_name, _, avg_train_loss, edev_loss, dev_acc, dev_f1))
            
            score = (dev_acc+dev_f1)/2
            if score>best_dev_score: # compute result for test part and store to out file, if we found better model
                best_dev_score = score
                cv_res[fold] = (dev_acc, dev_f1)

                predictions, true_labels = [], []
                for batch in test_dataloader:
                    batch = tuple(t.to(device) for t in batch)
                    b_input_ids, b_input_mask, b_labels = batch

                    with torch.no_grad():
                        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
                    logits = outputs[1].detach().cpu().numpy()
                    label_ids = b_labels.to('cpu').numpy()
                    predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
                    true_labels.extend(label_ids)
                restored_labels, restored_pred = untokenize(idx2tag, tst_inputs, tst_word_indices, predictions, true_labels)
                
                out_fname = os.path.join(out_dir, task_name+'.jsonl')
                with open(out_fname, 'w') as fw:
                    for idx, labels, prediction in zip(all_ids[test_mask], restored_labels, restored_pred):
                        data = {index_id:idx, label_id:labels, 'prediction':prediction}
                        json.dump(data, fw, ensure_ascii=False)
                        fw.write('\n')
        del model; gc.collect(); torch.cuda.empty_cache()

    dev_acc, dev_f1 = cv_res[0]
    print ('\ntask scores {}: {:.2f}/{:.2f}'.format(task_name, dev_acc, dev_f1))
