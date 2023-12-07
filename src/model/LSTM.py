from torch import Tensor
import torch.nn as nn
from easydict import EasyDict


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


def get_model(config: EasyDict) -> LSTM:
    """ get LSTM model according a configuration """
    model = LSTM(num_word=33992,
                 embedding_size=config.model.embedding_size,
                 hidden_layer_size=config.model.hidden_size,
                 num_classes=config.model.num_classes)
    return model