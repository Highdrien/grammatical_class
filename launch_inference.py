
import torch
from src.dataloader.vocabulary import load_dictionary,replace_word2int
from src.model.get_model import get_model

import os
import json

import yaml
from easydict import EasyDict
from typing import List, Dict

from src.train import train
from config.process_config import process_config


def load_config(path: str) -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def load_dictionary(path):
    with open(path, 'r') as f:
        dictionary = json.load(f)
    return dictionary

# Usage
dictionary = load_dictionary('path_to_your_json_file.json')


def main() -> None:
    config = load_config(os.path.join('config', 'config.yaml'))
    process_config(config)
    train(config)

def find_classes(tenseur):
    """
    Finds the classes with the highest probability for each element in the tensor.

    Args:
        tenseur (torch.Tensor): The input tensor.

    Returns:
        list: A list of lists containing the indices of the classes with the highest probability for each element in the tensor.
    """

    # Appliquer la fonction argmax sur le dernier axe du tenseur
    indices_max = torch.argmax(tenseur, dim=-1)
    
    # Convertir le tenseur d'indices en une liste de listes d'indices
    indices_max_liste = indices_max.squeeze().tolist()

    return indices_max_liste

def find_POS(list_index):
    """
    Finds the part-of-speech (POS) classes corresponding to the given list of indices.
    
    Args:
        list_index (list): A list of indices representing the POS classes.
        
    Returns:
        list: A list of POS classes corresponding to the given indices.
    """
    
    POS_CLASSES = ['PRON', 'AUX', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PROPN', 'NUM', 'CCONJ',
               '<PAD>', 'ADV', 'PART', 'INTJ', 'SYM', 'PUNCT', 'SCONJ', 'X', '_']
    POS = []
    for index in list_index:
        POS.append(POS_CLASSES[index])
    return POS
    



def convert_sentence_to_indexes(sentence,dictionary):
    """ convert a sentence to a list of indexes
    sentence: list of string
    dictionary: dict[string, int]
    """
    indexes = []
    for word in sentence:
        if word in dictionary:
            indexes.append(dictionary[word])
        else:
            indexes.append(dictionary['<UNK>'])
    return indexes


def inference(sentence: List[str],
              dictionary: Dict[str, int],
              experiment_path: str
              ) -> List[str]:
    """ run the model on a sentence
    sentence: list of string
    dictionary: dict[string, int]
    experiment_path: chemin de l'experience: exemple: logs/get_pos_lstm_2
    return: list of POS labels
    """
    #load config
    config = load_config(os.path.join(experiment_path, 'config.yaml'))
    process_config(config)

    #load model
    model = get_model(config)

    #load weights
    WEIGHT_PATH = os.path.join(experiment_path, 'checkpoint.pt')
    model.load_state_dict(torch.load(WEIGHT_PATH))
    del WEIGHT_PATH

    #find device
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")

    #prepare sentence 
    indexes = convert_sentence_to_indexes(sentence,dictionary)
    indexes = torch.tensor(indexes)
    indexes = indexes.unsqueeze(0).to(device)

    output = model(indexes)
    #take this argmax for each output element 

    if 'pos' in experiment_path: #if the model is a pos model
        output = find_classes(output)
        #convert indexes to POS
        output = find_POS(output)

    if 'morphy' in experiment_path:
        pass


    return output



if __name__ == '__main__':
    #load a dictionary
    dictionary = load_dictionary("dictionary/English.json")
    experiment_path = "logs/get_pos_lstm_3"
    #sentence in english
    sentence = ['i', 'need', 'a','dog', '.','But it aint','gonna','happen','.','I','dont','have','the','money','.']
    #sentence in french
    #sentence = ['je', 'veux', 'un','chien', '.','Mais ça','narrivera','pas','.','Je','nai','pas','dargent','.']
    #output: [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 1]
    res=inference(sentence, dictionary, experiment_path)
    print("sentence:",sentence)
    print("POS:",res)
