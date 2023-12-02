# procces data

import os
from easydict import EasyDict
from icecream import ic 
from typing import List

import get_sentences
import get_sequences
import vocabulary


Word_Info = List[str]       # example: ['6', 'flights', 'flight', 'NOUN', '_', 'Number=Plur', '1', 'obj', '_', '_']
Sentence = List[Word_Info]  # list of word info
Sequence = List[Word_Info]  # list of word info with the same length


def get_data(cfg: EasyDict, mode: str) -> List[Sequence]:
    folders = get_sentences.get_foldersname_from_language(datapath=cfg.path,
                                                          language=cfg.language)
    files = get_sentences.get_file_for_mode(folder_list=folders, mode=mode)
    data = get_sentences.get_sentences(files=files, indexes=cfg.indexes)

    word_index = get_sentences.get_word_index_in_indexes(cfg.indexes)
    vocab_path = os.path.join(cfg.vocab.path, cfg.language + '.json')
    
    if cfg.vocab.save and mode == 'train':
        vocab = vocabulary.create_vocab(data, word_index=word_index, pad=cfg.pad, unk=cfg.unk)
        vocabulary.save_vocab(vocab, path=vocab_path)
    else:
        vocab_path = os.path.join(cfg.vocab.path, cfg.language + '.json')
        vocab = vocabulary.load_dictionary(filepath=vocab_path)
    
    sequence_function = get_sequences.find_sequence_function(cfg.sequence_function)
    data = get_sequences.create_sequences(sentences=data,
                                          sequence_function=sequence_function,
                                          k=cfg.sequence_length,
                                          pad=cfg.pad)
    
    vocabulary.replace_word2int(data,
                                word_index=word_index,
                                vocab=vocab,
                                unk_rate=cfg.vocab.unk_rate,
                                unk=cfg.unk)

    return data


if __name__ == '__main__':
    # must to run dataloader like python .\src\dataloader\dataloader.py
    import yaml
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))
    ic(config)
    data = get_data(cfg=config.data, mode='train')
    for i in range(4):
        ic(i, data[i])
    
   
    