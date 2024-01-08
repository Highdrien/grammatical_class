# 1. Dictionnaire : mot -> premier label rencontré
# 2. Attribuer les labels à chaque mot de la séquence
# Si mot OOV -> UNK

import torch
from typing import List
from icecream import ic
from easydict import EasyDict
from src.dataloader.vocabulary import save_vocab

import src.dataloader.get_sentences as get_sentences

Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sequence = List[Word_Info]  # list of word info with the same length
Sentence = List[Word_Info]  # list of word info


def create_dictionnaire(data: list[Sentence]) -> dict[str,str]:
    """
    Crée un dictionnaire des mots de toutes les phrases de data associés à leur premier label rencontré
    """
    dico = {}
    for sentence in data:
        for word in sentence:
            if word[0] not in dico:
                dico[word[0]] = word[1]
    save_vocab(dico, "dictionary/baseline.json")
    return dico


def prediction(dico: dict[str,str], sequence: list[Sentence]) -> list[str]:
    """
    Réalise une prédiction des classes des mots d'une séquence de sentences, 
    en attribuant au mot la classe correspondante dans le dictionnaire. 
    Si le mot est OOV, alors la classe attribuée est UNK.
    """
    prediction = []
    for sentence in sequence:
        for word in sentence:
            if word in dico:
                prediction.append(dico[word])
            else:
                prediction.append("<UNK>")
    return prediction


def launch_baseline(config: EasyDict) -> None:
    """
    Lance le modèle baseline
    """
    folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
    files = get_sentences.get_file_for_mode(folder_list=folders, mode="train")
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])
    create_dictionnaire(data=data)


if __name__ == "__main__":
    config = EasyDict()
    launch_baseline(config=config)