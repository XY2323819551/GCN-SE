import numpy as np
import torch
import torch.nn as nn
# from pytorch_transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel


class GCNNetwork(nn.Module):
    def __init__(self, hidden_size):
        super(GCNNetwork, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = nn.Tanh()

    def forward(self, hidden, mask_matrix):
        normed_mask = mask_matrix.float()
        hidden = self.fc1(torch.matmul(normed_mask, hidden))
        outputs = self.activation(hidden)
        nozero_nums = (outputs != 0).sum(1)
        outputs = torch.sum(outputs, 1)
        avg_output = torch.div(outputs, nozero_nums)
        return avg_output.type_as(hidden)


class GCNNetwork2(nn.Module):
    def __init__(self, hidden_size):
        super(GCNNetwork2, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        # self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.activation = nn.Tanh()

    def forward(self, hidden, mask_matrix):
        normed_mask = mask_matrix.float()
        hidden = self.fc1(torch.matmul(normed_mask, hidden))
        outputs = self.activation(hidden)
        nozero_nums = (outputs != 0).sum(1)
        outputs = torch.sum(outputs, 1)
        avg_output = torch.div(outputs, nozero_nums)
        return avg_output.type_as(hidden)


def merge(a, b):
    return torch.cat([a, b, torch.abs(a - b)], 1)


def get_valid_embeddings(sequence_output, valid_ids, mem_valid_ids, device_type):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device_type)

    for i in range(batch_size):
        temp = sequence_output[i][valid_ids[i] == 1]
        valid_output[i][:temp.size(0)] = temp
    mem_valid_ids = torch.unsqueeze(mem_valid_ids, 2)
    valid_output = valid_output * mem_valid_ids
    return valid_output


class Bert(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.incro = config.incro

        self.bert = BertModel(self.config)
        # self.bert_dropout = nn.Dropout(config.bert_dropout)
        self.reg_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, self.num_labels),
            nn.Softmax(dim=-1),
        )

    def forward(self, input_ids, attention_masks, input_ids_b, attention_masks_b):
        # ??????A
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_masks,
                            output_hidden_states=True, return_dict=True)
        # print(outputs)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.incro == 'cls':
            c1 = pooled_output
            # print('using cls vector')
        elif self.incro == 'last-avg':
            c1 = (sequence_output * attention_masks.unsqueeze(-1)).sum(1) / attention_masks.sum(-1).unsqueeze(-1)

        # ??????B
        outputs = self.bert(input_ids=input_ids_b, attention_mask=attention_masks_b,
                            output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.incro == 'cls':
            c2 = pooled_output
        elif self.incro == 'last-avg':
            c2 = (sequence_output * attention_masks_b.unsqueeze(-1)).sum(1) / attention_masks_b.sum(-1).unsqueeze(-1)

        c = merge(c1, c2)
        logits = self.reg_layer(c)
        return logits


class Roberta(RobertaPreTrainedModel):
    def __init__(self, config):
        super(Roberta, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.incro = config.incro

        self.roberta = RobertaModel(self.config)
        # self.roberta_dropout = nn.Dropout(config.bert_dropout)
        self.reg_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size * 3, self.num_labels),
            nn.Softmax(dim=-1),
        )

    def forward(self, input_ids, attention_masks, input_ids_b, attention_masks_b):
        # ??????A
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_masks,
                               output_hidden_states=True, return_dict=True)
        # print(outputs)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        # hidden_states = outputs.hidden_states

        # pooled_output = self.roberta_dropout(pooled_output)
        if self.incro == 'cls':
            c1 = pooled_output
            # print('using cls vector')
        elif self.incro == 'last-avg':
            c1 = (sequence_output * attention_masks.unsqueeze(-1)).sum(1) / attention_masks.sum(-1).unsqueeze(-1)

        # ??????B
        outputs = self.roberta(input_ids=input_ids_b, attention_mask=attention_masks_b,
                               output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        if self.incro == 'cls':
            c2 = pooled_output
        elif self.incro == 'last-avg':
            c2 = (sequence_output * attention_masks_b.unsqueeze(-1)).sum(1) / attention_masks_b.sum(-1).unsqueeze(-1)

        c = merge(c1, c2)
        logits = self.reg_layer(c)
        return logits


class Bert_gcn(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_gcn, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.device_type = config.device
        self.incro = config.incro
        # self.alpha = config.alpha

        self.bert = BertModel(self.config)
        # self.bert_dropout = nn.Dropout(config.bert_dropout)
        self.reg_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size * 6, self.num_labels),
            nn.Softmax(dim=-1),
        )

        self.gcn = GCNNetwork(config.hidden_size)
        # self.gcn2 = GCNNetwork2(config.hidden_size)

    def forward(self, input_ids, attention_masks, valid_ids, mem_valid_ids, dep_adj_matrix,
                input_ids_b, attention_masks_b, valid_ids_b, mem_valid_ids_b, dep_adj_matrix_b):
        # ??????A
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_masks,
                            output_hidden_states=True, return_dict=True)
        # print(outputs)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        # hidden_states = outputs.hidden_states
        # cls_hidden = sequence_output[:, 0]
        # print(len(input_ids[0]))
        # print('cls shape', pooled_output.shape)
        # print('last layer shape:', sequence_output.shape)

        valid_output = get_valid_embeddings(sequence_output, valid_ids, mem_valid_ids, self.device_type)
        # gcn
        o1 = self.gcn(valid_output, dep_adj_matrix)

        # pooled_output = self.bert_dropout(pooled_output)

        if self.incro == 'mean':
            c1 = (pooled_output + o1) / 2.0
        elif self.incro == 'add':
            c1 = pooled_output + o1
        elif self.incro == 'weighted-avg':
            c1 = (1 - self.alpha) * pooled_output + self.alpha * o1
        elif self.incro == 'linear_avg':  # 08-06,new fusion method
            c1 = pooled_output + self.alpha * o1
        elif self.incro == 'concat':
            c1 = torch.cat((pooled_output, o1), 1)
        elif self.incro == 'token-vec':
            c1 = o1

        # ??????B
        outputs = self.bert(input_ids=input_ids_b, attention_mask=attention_masks_b,
                            output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        valid_output_b = get_valid_embeddings(sequence_output, valid_ids_b, mem_valid_ids_b, self.device_type)

        # gcn
        o2 = self.gcn(valid_output_b, dep_adj_matrix_b)

        if self.incro == 'mean':
            c2 = (pooled_output + o2) / 2.0
        elif self.incro == 'add':
            c2 = pooled_output + o2
        elif self.incro == 'weighted-avg':
            c2 = (1 - self.alpha) * pooled_output + self.alpha * o2
        elif self.incro == 'linear_avg':
            c2 = pooled_output + self.alpha * o2
        elif self.incro == 'concat':
            c2 = torch.cat((pooled_output, o2), 1)
        elif self.incro == 'token-vec':
            c2 = o2

        c = merge(c1, c2)
        logits = self.reg_layer(c)
        return logits


class Roberta_gcn(RobertaPreTrainedModel):
    def __init__(self, config):
        super(Roberta_gcn, self).__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.device_type = config.device
        self.incro = config.incro
        # self.alpha = config.alpha

        self.roberta = RobertaModel(self.config)
        self.reg_layer = nn.Sequential(
            nn.Linear(self.config.hidden_size * 6, self.num_labels),
            nn.Softmax(dim=-1),
        )

        self.gcn = GCNNetwork(config.hidden_size)

    def forward(self, input_ids, attention_masks, valid_ids, mem_valid_ids, dep_adj_matrix,
                input_ids_b, attention_masks_b, valid_ids_b, mem_valid_ids_b, dep_adj_matrix_b):
        # ??????A
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_masks,
                               output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        valid_output = get_valid_embeddings(sequence_output, valid_ids, mem_valid_ids, self.device_type)

        # gcn
        o1 = self.gcn(valid_output, dep_adj_matrix)
        if self.incro == 'mean':
            c1 = (pooled_output + o1) / 2.0
        elif self.incro == 'add':
            c1 = pooled_output + o1
        elif self.incro == 'weighted-avg':
            c1 = (1 - self.alpha) * pooled_output + self.alpha * o1
        elif self.incro == 'concat':
            c1 = torch.cat((pooled_output, o1), 1)
        elif self.incro == 'token-vec':
            c1 = o1

        # ??????B
        outputs = self.roberta(input_ids=input_ids_b, attention_mask=attention_masks_b,
                               output_hidden_states=True, return_dict=True)
        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        valid_output_b = get_valid_embeddings(sequence_output, valid_ids_b, mem_valid_ids_b, self.device_type)

        # gcn
        o2 = self.gcn(valid_output_b, dep_adj_matrix_b)

        if self.incro == 'mean':
            c2 = (pooled_output + o2) / 2.0
        elif self.incro == 'add':
            c2 = pooled_output + o2
        elif self.incro == 'weighted-avg':
            c2 = (1 - self.alpha) * pooled_output + self.alpha * o2
        elif self.incro == 'concat':
            c2 = torch.cat((pooled_output, o2), 1)
        elif self.incro == 'token-vec':
            c2 = o2

        c = merge(c1, c2)
        logits = self.reg_layer(c)
        return logits
