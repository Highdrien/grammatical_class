from torch import Tensor
import torch.nn as nn
from easydict import EasyDict
from transformers import BertModel


class LSTM(nn.Module):
    def __init__(self,
                 num_word: int,
                 embedding_size: int,
                 hidden_layer_size: int,
                 num_classes: int
                 ) -> None:
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(num_word, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_layer_size, batch_first=True)
        self.fc = nn.Linear(hidden_layer_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """ forward
        take x Tensor with shape: (B, K)
        return output Tensor with shape: (B, C, K)
        where:
            B: batch_size
            K: sequence size
            C: number of classes
        """
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        return output
    
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_size * 2, output_size)  # Multiply hidden_size by 2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.relu(lstm_out)
        output = self.fc(lstm_out)
        
        return output
    

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




def get_model(config: EasyDict) -> LSTM:
    """ get LSTM model according a configuration """
    model = LSTM(num_word=33992,
                 embedding_size=config.model.embedding_size,
                 hidden_layer_size=config.model.hidden_size,
                 num_classes=config.model.num_classes)
    return model