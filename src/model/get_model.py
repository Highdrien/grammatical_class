import torch.nn as nn
from easydict import EasyDict

from src.model.LSTM import LSTMClassifier
from src.model.BERT import BertClassifier


def get_model(config: EasyDict) -> nn.Module:
    """ get LSTM model according a configuration """

    if config.model.model_name == "lstm":
        lstmconfig = config.model.lstm
        model = LSTMClassifier(num_words=33992,
                               embedding_size=lstmconfig.embedding_size,
                               lstm_hidd_size_1=lstmconfig.lstm_hidd_size_1,
                               lstm_hidd_size_2=lstmconfig.lstm_hidd_size_2,
                               fc_hidd_size=lstmconfig.fc_hidd_size,
                               num_classes=config.model.num_classes,
                               bidirectional=lstmconfig.bidirectional,
                               activation=lstmconfig.activation)
        
    elif config.model.model_name == "bert":
        bertconfig = config.model.bert
        model = BertClassifier(pretrained_model_name=bertconfig.pretrained_model_name,
                               hidden_size=bertconfig.hidd_size,
                               num_classes=config.model.num_classes)
    
    return model
