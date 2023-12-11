from torch import Tensor
import torch.nn as nn
from easydict import EasyDict
from numpy import prod
from transformers import BertModel

from typing import List, Union
import torch
from icecream import ic


class BertClassifier(nn.Module):
    def __init__(self, pretrained_model_name, hidden_size, num_classes): #exemple name: 'bert-base-uncased' 'distilcamembert-base-ner'
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name) #pas nécéssaire de spécifier l'input shape
        self.dropout = nn.Dropout(0.1)  # You can adjust the dropout rate
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids):
        attention_mask = None #torch.zeros(input_ids.shape) 
        print("shape of input_ids:",input_ids.shape)
        print("first element of batch:",input_ids[0])
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits
    
    def get_number_parameters(self) -> int:
        """ return the number of parameters of the model """
        return sum([prod(param.size()) for param in self.parameters()])


if __name__ == '__main__':
    

    model = BertClassifier('bert-base-uncased', 768, 2)
    
    print(model)
    print(model.get_number_parameters())

    x = torch.randint(0, 10, (16, 12)) #représente une phrase de 12 mots, batch size de 16, 10 mots max par phrase
    print("shape entrée:",x.shape)

    y = model.forward(x)

    print("shape sortie",y.shape)

