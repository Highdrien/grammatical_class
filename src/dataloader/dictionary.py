# create dictionary of word and convert words to number with this dictionary

import json
from typing import Dict, List

Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]  # list of word info


def create_dico(data: List[Sentence], word_index: int, pad: str) -> Dict[str, int]:
    """ create a dictionary witch match each word to an unique number
    data: list of Sentences
    index: index of the word in the word_info
    pad: pad caracter (that will be in the dictionary if pad is not None)
    """
    dico = {}
    word_number = 0

    if pad is not None:
        dico[pad] = word_number
        word_number += 1
    
    for sentence in data:
        for word_info in sentence:
            word = word_info[word_index]
            if word not in dico:
                dico[word] = word_number
                word_number += 1
    return dico


def save_dictionary(dico: Dict[str, int], path: str) -> None:
    """ save the dictionary in the path"""
    with open(path, 'w') as f:
        json.dump(dico, f)
        f.close()


def load_dictionary(filename: str) -> Dict[str, int]:
    with open(filename, 'r') as f:
        dico = json.load(f)
        f.close()
    return dico
