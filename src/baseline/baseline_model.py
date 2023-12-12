# 1. Dictionnaire : mot -> premier label rencontré
# 2. Attribuer les labels à chaque mot de la séquence
# Si mot OOV -> UNK

from icecream import ic

import src.dataloader.get_sentences as get_sentences
import src.dataloader.get_sequences as get_sequences
import src.dataloader.vocabulary as vocabulary
import src.dataloader.convert_label as convert_label
import src.dataloader.get_word_and_label as get_word_and_label

folders = get_sentences.get_foldersname_from_language(datapath="data", language="French")
files = get_sentences.get_file_for_mode(folder_list=folders, mode="train")
data = get_sentences.get_sentences(files=files, indexes=[1, 5])


def create_dictionnaire():
    """
    Crée un dictionnaire des mots associés à leur premier label rencontré
    """

    return 

if __name__ == "__main__":
    ic(data)