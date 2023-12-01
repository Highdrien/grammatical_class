# procces data

import os
from easydict import EasyDict
from icecream import ic 
from typing import List

import get_sentences
import get_sequences
import dictionary


Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]  # list of word info
Sequence = List[Word_Info]  # list of word info with the same length


def get_data(cfg: EasyDict, mode: str) -> List[Sequence]:
    folders = get_sentences.get_foldersname_from_language(datapath=cfg.path,
                                                          language=cfg.language)
    files = get_sentences.get_file_for_mode(folder_list=folders, mode=mode)
    data = get_sentences.get_data(files=files, indexes=cfg.indexes)
    word_index = get_sentences.get_word_index_in_indexes(cfg.indexes)
    if mode == 'train':
        dico = dictionary.create_dico(data, word_index=word_index, pad=cfg.pad)
        dico_path = os.path.join(cfg.dicopath, cfg.language + '.json')
        dictionary.save_dictionary(dico, path=dico_path)
    sequence_function = get_sequences.find_sequence_function(cfg.sequence_function)
    data = get_sequences.create_sequences(sentences=data,
                                          sequence_function=sequence_function,
                                          k=cfg.sequence_length,
                                          pad=cfg.pad)
    return data


if __name__ == '__main__':
    config = {'path': os.path.join('..', '..', 'data'),
              'language': 'English',
              'sequence_length': 10,
              'pad': '<PAD>',
              'sequence_function': 'dummy',
              'indexes': [1, 3],
              'dicopath': os.path.join('..', '..', 'dictionary')}
    ic(config)
    data = get_data(cfg=EasyDict(config), mode='train')
    for i in range(4):
        ic(i, data[i])
    
   
    