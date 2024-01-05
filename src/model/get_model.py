import os
import torch
from easydict import EasyDict

from typing import Optional

from src.model.LSTM import LSTMClassifier, MorphLSTMClassifier
# from src.model.BERT import BertClassifier


def get_model(config: EasyDict) -> torch.nn.Module:
    """ get model according a configuration """

    task_name = config.task.task_name
    num_classes = config.task[f'{task_name}_info'].num_classes

    if task_name == "get_pos":
        pos_config = config.model.lstm_pos
        model = LSTMClassifier(num_words=config.data.vocab.num_words,
                               embedding_size=pos_config.embedding_size,
                               lstm_hidd_size_1=pos_config.lstm_hidd_size_1,
                               lstm_hidd_size_2=pos_config.lstm_hidd_size_2,
                               fc_hidd_size=pos_config.fc_hidd_size,
                               num_classes=num_classes,
                               bidirectional=pos_config.bidirectional,
                               activation=pos_config.activation,
                               num_c_possibility=1,
                               dropout=pos_config.dropout)
    
    elif task_name == 'get_morphy':
        if config.model.model_name == 'lstm':
            morphy_config = config.model.lstm_morphy
            model = LSTMClassifier(num_words=config.data.vocab.num_words,
                                   embedding_size=morphy_config.embedding_size,
                                   lstm_hidd_size_1=morphy_config.lstm_hidd_size_1,
                                   lstm_hidd_size_2=morphy_config.lstm_hidd_size_2,
                                   fc_hidd_size=morphy_config.fc_hidd_size,
                                   num_classes=num_classes,
                                   bidirectional=morphy_config.bidirectional,
                                   activation=morphy_config.activation,
                                   num_c_possibility=config.task.get_morphy_info.num_features,
                                   dropout=morphy_config.dropout)
            
        if config.model.model_name == 'lstm_separate':
            morphy_config = config.model.lstm_morphy
            model = MorphLSTMClassifier(num_words=config.data.vocab.num_words,
                                   embedding_size=morphy_config.embedding_size,
                                   lstm_hidd_size_1=morphy_config.lstm_hidd_size_1,
                                   lstm_hidd_size_2=morphy_config.lstm_hidd_size_2,
                                   fc_hidd_size=morphy_config.fc_hidd_size,
                                   num_classes=num_classes,
                                   bidirectional=morphy_config.bidirectional,
                                   activation=morphy_config.activation,
                                   num_c_possibility=config.task.get_morphy_info.num_features)
        
        else:
            raise NotImplementedError("No other lstm model for get morphy was implemented")
    
    else:
        raise NotImplementedError(f"Error, expected task_name be get_morphy or get_pos but found: {task_name}")
    
    return model


def load_checkpoint(model: torch.nn.Module,
                    experiment_path: str,
                    device: Optional[torch.device]='cpu') -> None:
    """ load checkpoint from logging_path in the model """
    weight_path = os.path.join(experiment_path, 'checkpoint.pt')

    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"weigth was not found in {weight_path}")
    
    model.load_state_dict(torch.load(weight_path, map_location=device))

