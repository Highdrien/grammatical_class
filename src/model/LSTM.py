from torch import Tensor
import torch.nn as nn
from numpy import prod

from typing import List, Union


class LSTMClassifier(nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_size: int,
                 lstm_hidd_size_1: int,
                 lstm_hidd_size_2: Union[int, None],
                 fc_hidd_size: List[int],
                 num_classes: int,
                 bidirectional: bool,
                 activation: str
                 ) -> None:
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_words,
                                      embedding_dim=embedding_size)
        
        # LSTM Layers
        mul = 2 if bidirectional else 1
        self.lstm_1 = nn.LSTM(input_size=embedding_size,
                              hidden_size=lstm_hidd_size_1,
                              batch_first=True,
                              bidirectional=bidirectional)

        self.have_lstm_2 = lstm_hidd_size_2 is not None
        if self.have_lstm_2:
            self.lstm_2 = nn.LSTM(input_size=lstm_hidd_size_1 * mul,
                                  hidden_size=lstm_hidd_size_2,
                                  batch_first=True,
                                  bidirectional=bidirectional)
        
        assert activation in ['relu', 'sigmoid'], f"Error, activation must be relu or sigmoid but found '{activation}'"
        self.activation = nn.ReLU() if activation == 'relu' else nn.Sigmoid()

        # Fully Connected Layers
        last_lstm_hidd_layers_size = lstm_hidd_size_1 if not self.have_lstm_2 else lstm_hidd_size_2
        fc_hidd_size = [last_lstm_hidd_layers_size * mul] + fc_hidd_size + [num_classes]
        self.fc = []

        for i in range(len(fc_hidd_size) - 1):
            self.fc.append(self.activation)
            self.fc.append(nn.Linear(in_features=fc_hidd_size[i],
                                     out_features=fc_hidd_size[i + 1]))
        self.fc = nn.Sequential(*self.fc)

    def forward(self, x: Tensor) -> Tensor:
        """ forward
        take x Tensor with shape: (B, K)
        return output Tensor with shape: (B, K, C)

        where:
            B: batch_size
            K: sequence size
            C: number of classes
        """
        x = self.embedding(x)
        x = self.activation(x)

        x, _ = self.lstm_1(x)

        if self.have_lstm_2:
            x, _ = self.lstm_2(x)
        
        logits = self.fc(x)
        return logits
    
    def get_number_parameters(self) -> int:
        """ return the number of parameters of the model """
        return sum([prod(param.size()) for param in self.parameters()])
    

if __name__ == '__main__':
    import torch
    from icecream import ic

    model = LSTMClassifier(num_words=10,
                           embedding_size=32,
                           lstm_hidd_size_1=64,
                           lstm_hidd_size_2=32,
                           fc_hidd_size=[64],
                           num_classes=6,
                           bidirectional=True,
                           activation='sigmoid')
    
    ic(model)
    ic(model.get_number_parameters())

    x = torch.randint(0, 10, (16, 12))
    ic(x.shape)

    y = model.forward(x)

    # ic(y)
    ic(y.shape)

