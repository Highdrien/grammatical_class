# Prendre toutes les phrases de test, et calculer l'accuracy du modèle baseline

import torch
from typing import List
from icecream import ic
from easydict import EasyDict
import baseline_dictionary
import src.metrics
from src.dataloader import get_sentences, convert_label


def test_dictionary(test_path: str, baseline_path: str) -> dict[str,str]:
    """
    Crée un dictionnaire des mots de toutes les phrases de données de test associés à leur premier label rencontré
    """
    folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
    files = get_sentences.get_file_for_mode(folder_list=folders, mode="test")
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])  # indexes for morphy
    dico = baseline_dictionary.create_dictionnaire(data_path="data", language="French", mode="train")
    
    
    metrics = get_metrics(config=config, device=device)
    metrics_name = metrics.get_metrics_name()
    test_metrics = np.zeros((len(metrics_name)))
    count_unk = 0
    for sentence in data:
        for word in sentence:
            if word[0] in dico:
                if dico[word[0]] == word[1]:
                    test_loss += loss.item()
            test_metrics += metrics.compute(y_true=y_true, y_pred=y_pred)
                    count_accuracy += 1
            else:
                if word[1] == "<UNK>":
                    count_unk += 1
    print(f'{count_accuracy = }')


    test_metrics = test_metrics / n_test
    return test_metrics


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