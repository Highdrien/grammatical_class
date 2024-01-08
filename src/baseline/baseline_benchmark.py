# Prendre toutes les phrases de test, et calculer l'accuracy du modèle baseline

import torch
import numpy as np
from typing import List
from icecream import ic
from easydict import EasyDict
import baseline_dictionary
from src.metrics import MOR_Metrics
from src.dataloader import get_sentences, convert_label


def test_dictionary(test_path: str, baseline_path: str) -> dict[str,str]:
    """
    Crée un dictionnaire des mots de toutes les phrases de données de test associés à leur premier label rencontré
    """
    folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
    files = get_sentences.get_file_for_mode(folder_list=folders, mode="test")
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])  # indexes for morphy
    # faire un get sequence
    dico_train = baseline_dictionary.create_dictionnaire(data_path="data", language="French", mode="train")
    
    config = {'data': {'sequence_length': 10},
              'task': {'get_morphy_info': {'num_classes': 28},
                       'task_name': 'get_morphy'},
              'metrics': {'acc': True, 'allgood': True} }
    
    metrics = MOR_Metrics(config=EasyDict(config))
    metrics_name = metrics.get_metrics_name()
    baseline_metrics = np.zeros((len(metrics_name)))

    label_encoder = convert_label.Morphy()
    unk_label = '_=Yes'

    n = 0

    for sentence in data:

        y_pred_sequence = []
        y_true_sequence = []

        for word, y_true in sentence:
            n += 1 
            y_pred = dico_train[word] if word in dico_train else unk_label

            y_pred_sequence.append(label_encoder.encode(label_to_convert=y_pred))
            y_true_sequence.append(label_encoder.encode(label_to_convert=y_true))
        
        

        y_true_encoding = torch.tensor([y_true_sequence])
        y_pred_encoding = torch.tensor([y_pred_sequence])

        # les faire passer en one-hot dim: -1
        
        # y_pred et y_true de shape 1, 10, 28, 13
        baseline_metrics += metrics.compute(y_true=y_true_encoding, y_pred=y_pred_encoding)
    

    baseline_metrics = baseline_metrics / n
    return baseline_metrics


# 


def test_prediction(dico: dict[str,str], sequence: list[Sentence]) -> float:
    """
    Réalise une prédiction des classes des mots d'une séquence de sentences, 
    en attribuant au mot la classe correspondante dans le dictionnaire. 
    Si le mot est OOV, alors la classe attribuée est UNK.
    """
    prediction = []
    for sentence in sequence:
        for word in sentence:
            if word in dico:
                if word[1] == dico[word]:
                    prediction.append(1)
                prediction.append(dico[word])
            else:
                prediction.append("<UNK>")
    


    print(f'{len(prediction) = }')
    return prediction


def launch_baseline(config: EasyDict) -> None:
    """
    Lance le modèle baseline
    """
    create_dictionnaire(data_path=config.data.path, language=config.data.language, mode=config.data.mode)


if __name__ == "__main__":
    config = EasyDict()
    launch_baseline(config=config)