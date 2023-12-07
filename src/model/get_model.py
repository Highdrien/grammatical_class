import torch.nn as nn
from easydict import EasyDict

from src.model.LSTM import LSTMClassifier
from src.model.BERT import BertClassifier


def get_model(config: EasyDict) -> nn.Module:
    """ get model according a configuration """

    task_name = config.task.task_name
    num_classes = config.task[f'{task_name}_info'].num_classes

    assert task_name == 'get_pos', NotImplementedError(f"only the task get_pos was implemended")

    if config.model.model_name == "lstm":
        lstmconfig = config.model.lstm
        model = LSTMClassifier(num_words=config.data.vocab.num_words,
                               embedding_size=lstmconfig.embedding_size,
                               lstm_hidd_size_1=lstmconfig.lstm_hidd_size_1,
                               lstm_hidd_size_2=lstmconfig.lstm_hidd_size_2,
                               fc_hidd_size=lstmconfig.fc_hidd_size,
                               num_classes=num_classes,
                               bidirectional=lstmconfig.bidirectional,
                               activation=lstmconfig.activation)
        
    if config.model.model_name == "bert":
        bertconfig = config.model.bert
        model = BertClassifier(pretrained_model_name=bertconfig.pretrained_model_name,
                               hidden_size=bertconfig.hidd_size,
                               num_classes=num_classes)
    
    return model
