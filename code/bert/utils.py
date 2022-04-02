# -*- coding: utf-8 -*-
import os
import random
import torch
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertConfig, BertPreTrainedModel, BertModel

class PoolBertForTokenClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, config.num_labels)
        
        self.w_size = 4
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs['last_hidden_state']
        
        shape = list(sequence_output.shape)
        shape[1]+=self.w_size-1

        t_ext = torch.zeros(shape, dtype=sequence_output.dtype, device=sequence_output.device)
        t_ext[:, self.w_size-1:, :] = sequence_output

        unfold_t = t_ext.unfold(1, self.w_size, 1).transpose(3,2)
        pooled_output_mean = torch.mean(unfold_t, 2)
        
        pooled_output, _ = torch.max(unfold_t, 2)
        pooled_output = torch.relu(pooled_output)
        
        sequence_output = torch.cat((pooled_output, pooled_output_mean, sequence_output), 2)
        
        sequence_output = self.dropout(sequence_output)
        
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss_mask = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)

                active_labels = torch.where(
                    active_loss_mask,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output

class PoolBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size*3, self.config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        
        encoder_out = outputs['last_hidden_state']
        cls = encoder_out[:, 0, :]
        
        pooled_output, _ = torch.max(encoder_out, 1)
        pooled_output = torch.relu(pooled_output)
        
        pooled_output_mean = torch.mean(encoder_out, 1)
        pooled_output = torch.cat((pooled_output, pooled_output_mean, cls), 1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                # We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss = F.binary_cross_entropy_with_logits( logits.view(-1), labels.view(-1) )
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
