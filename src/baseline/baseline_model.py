# 1. Dictionnaire : mot -> premier label rencontré
# 2. Attribuer les labels à chaque mot de la séquence
# Si mot OOV -> UNK

from icecream import ic
from easydict import EasyDict

import src.dataloader.get_sentences as get_sentences


def create_dictionnaire():
    """
    Crée un dictionnaire des mots associés à leur premier label rencontré
    """

    return 


def launch_baseline(config: EasyDict) -> None:
    print('Hello')




if __name__ == "__main__":
    folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
    files = get_sentences.get_file_for_mode(folder_list=folders, mode="train")
    data = get_sentences.get_sentences(files=files, indexes=[1, 5])
    ic(data)