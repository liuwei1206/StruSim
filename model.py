# author = liuwei
# date = 2021-10-18

import os
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

from layer import GraphConvolution
from transformers.modeling_bert import BertModel, BertLayerNorm
from transformers.modeling_xlnet import XLNetModel, XLNetLayerNorm, XLNetRelativeAttention
from transformers import PreTrainedModel


class Custom_GCN(nn.Module):
    """
    The Graph Convolution Network for node classification. This model is implemented by myself, instead of
    recalling the implementation of pytorch geometric
    """

    def __init__(self, params):
        super(Custom_GCN, self).__init__()

        self.input_dim = params['input_dim']
        self.hidden_dim = params['hidden_dim']
        self.input_dropout = nn.Dropout(params['input_dropout'])
        self.dropout = nn.Dropout(params['dropout'])
        self.featureless = params['featureless']
        self.num_labels = params['num_labels']
        self.num_gc_layer = params['num_gc_layer']

        total_layers = []
        if self.num_gc_layer == 1:
            total_layers.append(GraphConvolution(self.input_dim, self.num_labels, featureless=self.featureless))
        else:
            total_layers.append(GraphConvolution(self.input_dim, self.hidden_dim, featureless=self.featureless))
            for idx in range(1, self.num_gc_layer - 1):
                total_layers.append(GraphConvolution(self.hidden_dim, self.hidden_dim))
            total_layers.append(GraphConvolution(self.hidden_dim, self.num_labels))

        self.layers = nn.ModuleList(total_layers)

    def forward(self, features, adjacency_matrix, mask=None, labels=None, flag="Train"):
        """
        Args:
            features: the representation of each nodes
            adjacency_matrix:
            mask: to differ train, dev, test
            labels: train labels, dev labels, test labels
        """
        all_hidden_states = (features,)
        features = self.input_dropout(features)
        for idx, layer in enumerate(self.layers):
            features, adjacency_matrix = layer((features, adjacency_matrix))
            if idx + 1 < self.num_gc_layer:
                features = F.relu(features)
                features = self.dropout(features)
            all_hidden_states = all_hidden_states + (features,)

        logits = features
        _, preds = torch.max(logits, dim=-1)
        outputs = (preds, all_hidden_states)
        if flag.upper() == "TRAIN":
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            mask = mask.float()
            mask = mask / mask.mean()
            loss *= mask
            loss = loss.mean()
            outputs = (loss,) + outputs

        return outputs

    def l2_loss(self, applied_layers=[0]):
        loss = None
        for idx in applied_layers:
            layer = self.layers[idx]
            for param in layer.parameters():
                if loss:
                    loss += param.pow(2).sum()
                else:
                    loss = param.pow(2).sum()

        return loss


class Graphlet_DNN(nn.Module):
    """
    Graphlet features + DNN for classification task
    """

    def __init__(self, params):
        super(Graphlet_DNN, self).__init__()

        self.input_dim = params['input_dim']
        self.fc1_hidden_dim = params['fc1_hidden_dim']
        self.fc2_hidden_dim = params['fc2_hidden_dim']
        self.num_labels = params['num_labels']
        self.dropout = nn.Dropout(p=params['dropout'])

        self.fc1 = nn.Linear(self.input_dim, self.fc1_hidden_dim)
        self.fc2 = nn.Linear(self.fc1_hidden_dim, self.fc2_hidden_dim)
        self.classifier = nn.Linear(self.fc2_hidden_dim, self.num_labels)

    def forward(self, inputs, labels=None, flag="Train"):
        """
        Args:
            inputs: each sample is a vector
            labels:
            flag:
        """
        all_hidden_states = (inputs,)
        fc1_outputs = self.fc1(inputs)
        fc1_outputs = nn.Tanh()(fc1_outputs)
        fc1_outputs = self.dropout(fc1_outputs)
        all_hidden_states = all_hidden_states + (fc1_outputs,)
        fc2_outputs = self.fc2(fc1_outputs)
        fc2_outputs = nn.Tanh()(fc2_outputs)
        fc2_outputs = self.dropout(fc2_outputs)
        all_hidden_states = all_hidden_states + (fc2_outputs,)
        logits = self.classifier(fc2_outputs)
        all_hidden_states = all_hidden_states + (logits,)

        preds = torch.argmax(logits, dim=-1)
        outputs = (preds, all_hidden_states)
        if flag.upper() == "TRAIN":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class PretrainedModelForCohModeling(PreTrainedModel):
    """Pretrained model + classifier, then finetuning"""

    def __init__(self, config, params):
        super(PretrainedModelForCohModeling, self).__init__(config)

        self.num_labels = params['num_labels']
        self.hidden_dim = params['hidden_dim']
        self.freeze = params['freeze']
        model_path = params['model_path']
        self.encoder = XLNetModel.from_pretrained(model_path, config=config)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(config.hidden_size, self.hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.hidden_dim, self.num_labels)

        if self.freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.fc.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.fc.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.classifier.bias.data.zero_()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            flag="Train"
    ):
        if self.freeze:
            self.encoder.eval()
            with torch.no_grad():
                pretrained_outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
        else:
            pretrained_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )

        last_hidden_states = pretrained_outputs[0]
        batch_length = torch.sum(attention_mask, dim=-1)  # [batch]
        batch_sum = last_hidden_states * attention_mask.unsqueeze(2)  # [batch, seq_len, dim]
        batch_sum = torch.sum(batch_sum, dim=1)  # [batch, dim]
        pooling_outputs = batch_sum / batch_length.unsqueeze(1)  # [batch, dim]
        all_hidden_states = (pooling_outputs,)

        fc_outputs = self.fc(pooling_outputs)
        fc_outputs = self.relu(fc_outputs)
        all_hidden_states = all_hidden_states + (fc_outputs,)
        fc_outputs = self.dropout(fc_outputs)
        logits = self.classifier(fc_outputs)  # [N, C]
        all_hidden_states = all_hidden_states + (logits,)
        _, preds = torch.max(logits, dim=-1)
        # print(preds, labels)
        outputs = (preds, all_hidden_states)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='none')
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            loss = loss.mean()
            outputs = (loss,) + outputs

        return outputs
