
import torch
import numpy as np
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


def load_dictionary_morph(path):
    with open(path, 'r') as f:
        dictionary = json.load(f)
    return dictionary


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
    
def find_morphy(tab):
    """
    find the morphy using a numpy array containing 1 one hot vector per feature indicating its value
    dimension is: (nb_features, nb_classes)
    """
    #load dictionnary of morphy
    dictionary = load_dictionary_morph("src/dataloader/morphy.json")
    #get the list of the keys
    keys = list(dictionary.keys())
    #for each one hot vector of the tab
    morphy = []
    #get length of the tab
    length = tab.shape[0]
    list_keys= np.array(dictionary.keys())
    for k in range(length):
        #get the one hot vector
        one_hot = tab[k]
        #get the index of the 1
        index = torch.argmax(one_hot)
        #get the key corresponding to the index
        key = keys[k]
        #get the value corresponding to the key
        value = dictionary[key]
        #add the value to the list
        if index< len(value):
            morphy.append(value[index])
        else:
            #print("predicted index is out of range of the value list")
            morphy.append("ERROR")
    
    return list_keys, morphy
    



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
    return: list of POS labels / nothing if morphy
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

    if 'separate' in experiment_path or 'fusion' in experiment_path or 'supertag' in experiment_path: #if the model is a morphy model
        print("shape of output:",output.shape)

        for i in range(len(sentence)):
            print("WORD:",sentence[i])
            print("OUTPUT:",find_morphy(output[0][i])[1]) #on print seulement les valeurs pas la liste des clés
        return("done")
        


    return output



if __name__ == '__main__':
    #load a dictionary
    dictionary = load_dictionary("dictionary/French.json")
    #define the experiment path
    experiment_path = "logs/fusion"
    #define a sentence on which we want to do inference
    sentence = ['je', 'veux', 'un','chien', '.','Mais ça','narrivera','pas','.','Je','nai','pas','argent','.']
    #run inference
    res=inference(sentence, dictionary, experiment_path)
    print("sentence:",sentence)
    print("POS:",res)

