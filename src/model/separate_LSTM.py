import torch
from torch import Tensor
import torch.nn as nn
from numpy import prod

from typing import List, Union, Optional, Iterator, Tuple
from torch.nn.parameter import Parameter


class MorphLSTMClassifier(nn.Module):
    def __init__(self,
                 num_words: int,
                 embedding_size: int,
                 lstm_hidd_size_1: int,
                 num_classes: int,
                 lstm_hidd_size_2: Optional[Union[int, None]]=None,
                 fc_hidd_size: Optional[List[int]]=[], 
                 bidirectional: Optional[bool]=True,
                 activation: Optional[str]='relu',
                 num_c_possibility: Optional[int]=1,
                 dropout: Optional[float]=0
                 ) -> None:
        """ Model LSTM 
        ## Arguments:
        num_words: int
            number of word in the vocabulary
        embedding_size: int
            size of embedding
        lstm_hidd_size_1: int
            size of the first lstm layer
        num_classes: int
            number of classes

        ## Optional Arguments:
        lstm_hidd_size_2: int or None = None
            None -> not a second lstm layer; 
            int -> size of the second lstm layer
        fc_hidd_size: list of int = []
            List of size of Dense layers.
            !!! Not count the last layer in fc_hidd_size witch give the number of classes !!!
        bidirectional: bool = true
            Does the lstm layers go in both direction
        activation: str = relu
            choose an activation function between relu or softmax
        num_c_possibility: int = 1
            number of features of classes. must be 1 for get_pos and not 1 for get_mophy
        """
        super(MorphLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_words,
                                      embedding_dim=embedding_size)
        
        # LSTM Layers
        mul = 2 if bidirectional else 1
        self.lstm_1 = nn.LSTM(input_size=embedding_size,
                              hidden_size=lstm_hidd_size_1,
                              batch_first=True,
                              bidirectional=bidirectional)
        
        self.dropout = nn.Dropout(p=dropout)

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
        fc_hidd_size = [last_lstm_hidd_layers_size * mul] + fc_hidd_size + [num_classes * num_c_possibility]
        self.fc = []

        for i in range(len(fc_hidd_size) - 1):
            self.fc.append(self.dropout)
            self.fc.append(self.activation)
            self.fc.append(nn.Linear(in_features=fc_hidd_size[i],
                                     out_features=fc_hidd_size[i + 1]))
        self.fc = nn.Sequential(*self.fc)

        #Create a number of Dense layer equal to the number of classes num_classes and their size is num_c_possibility their input is the output of the last Dense layer
        self.morph: List[nn.Linear] = []
        for i in range(num_classes):
            self.morph.append(nn.Linear(in_features=num_c_possibility, out_features=num_c_possibility)) #chaque couche doit récupérer un object de taille 13
            

        self.do_morphy = (num_c_possibility != 1)
        self.num_classes = num_classes
        self.num_c_possibility = num_c_possibility

    def forward(self, x: Tensor) -> Tensor:
        """ forward
        take x Tensor with shape: (B, K)
        return output Tensor with shape: (B, K, C)

        where:
            B: batch_size
            K: sequence size
            C: number of classes
        """

        x = x.to("cuda")
        sequence_length = x.shape[-1]

        x = self.embedding(x)
        x = self.activation(x)

        x, _ = self.lstm_1(x)

        if self.have_lstm_2:
            x = self.dropout(x)
            x = self.activation(x)
            x, _ = self.lstm_2(x)

        logits = self.fc(x)

        if self.do_morphy:
            logits = logits.view(-1, sequence_length, self.num_classes, self.num_c_possibility)
            logit_list = []
            for i in range(self.num_classes):
                logits_i = self.morph[i](logits[:, :, i, :])
                logit_list.append(logits_i)
            logits = torch.stack(logit_list, dim=2)
            
        return logits

    def get_number_parameters(self) -> int:
        """ return the number of parameters of the model """
        return sum([prod(param.size()) for param in self.parameters()])

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from super().parameters(recurse=recurse)
        for c in range(self.num_classes):
            yield from self.morph[c].parameters(recurse=False)
    
    def named_parameters(self, prefix: str = '', recurse: bool = True, remove_duplicate: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield from super().named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
        for c in range(self.num_classes):
            yield from self.morph[c].named_parameters(prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate)
    
    def to(self, device: torch.device):
        self = super().to(device)
        for c in range(self.num_classes):
            self.morph[c] = self.morph[c].to(device)
        return self


if __name__ == '__main__':
    model = MorphLSTMClassifier(num_words=30,
                                embedding_size=64,
                                lstm_hidd_size_1=64,
                                num_classes=19,
                                bidirectional=True,
                                activation='relu',
                                num_c_possibility=13,
                                dropout=0.1)

    for name, param in model.named_parameters():
        print(name, param.shape)
        if param.shape == torch.Size([13, 13]):
            print(param[0, 0])

    
