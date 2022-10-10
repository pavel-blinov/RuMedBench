import json
import pathlib

import click
import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from utils import seed_everything, seed_worker

def encode_text_pairs(tokenizer, sentences):
    bs = 20000
    input_ids, attention_masks, token_type_ids = [], [], []
    
    for _, i in enumerate(range(0, len(sentences), bs)):
        tokenized_texts = []
        for sentence in sentences[i:i+bs]:
            final_tokens = ['[CLS]']+tokenizer.tokenize( sentence )[:MAX_LEN-2]+['[SEP]']
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
            token_type_row = np.zeros(row.shape[0], dtype=np.int)
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
                default='RuMedTest',
                type=click.Choice(['RuMedTest']),
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

    if device == -1:
        device = torch.device('cpu')
    else:
        device = torch.device(device)

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

    tokenizer = BertTokenizer.from_pretrained(
        out_path / 'models/rubert_cased_L-12_H-768_A-12_pt/',
        do_lower_case=True,
        max_length=MAX_LEN
    )

    from utils import BertFeatureExtractor as BertModel
    ## take appropriate config and init a BERT model
    config_path = out_path / 'models/rubert_cased_L-12_H-768_A-12_pt/bert_config.json'
    conf = BertConfig.from_json_file( config_path )
    model = BertModel(conf)
    ## preload it with weights
    output_model_file = out_path / 'models/rubert_cased_L-12_H-768_A-12_pt/pytorch_model.bin'
    model.load_state_dict(torch.load(output_model_file), strict=False)
    model = model.to(device)
    model.eval();

    def get_embeddings(texts):
        input_ids, attention_masks, token_type_ids = encode_text_pairs(tokenizer, texts)
        ##prediction_dataloader
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        token_type_ids = torch.tensor(token_type_ids)

        batch_size = 16
        prediction_data = TensorDataset(input_ids, attention_masks, token_type_ids)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size, worker_init_fn=seed_worker)

        predictions = []
        for step, batch in enumerate(prediction_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_token_type_ids = batch
            with torch.no_grad():
                outputs = model( b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, bert_type=bert_type )
                outputs = outputs.detach().cpu().numpy()
                predictions.append(outputs)
        predictions = expit(np.vstack(predictions))
        return predictions

    q_vecs = get_embeddings(test_data['question'])
    
    sims = []
    for option in options:
        option_vecs = get_embeddings(test_data[option])
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
