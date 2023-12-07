from torch import Tensor
import torch.nn as nn
from easydict import EasyDict
from numpy import prod
from transformers import BertModel

from typing import List, Union


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size, num_classes): #exemple name: 'bert-base-uncased' 'distilcamembert-base-ner'
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name) #pas nécéssaire de spécifier l'input shape
        self.dropout = nn.Dropout(0.1)  # You can adjust the dropout rate
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
