from typing import List, Callable


POS_CLASSES = ['PRON', 'AUX', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PROPN', 'NUM', 'CCONJ',
               '<PAD>', 'ADV', 'PART', 'INTJ', 'SYM', 'PUNCT', 'SCONJ', 'X', '_']


def convert_morphy(morphy: str) -> List[int]:
    """
    convert morphy like: Definite=Ind|PronType=Art by ...
    """
    raise NotImplementedError('convert morphy was not implemented')


def convert_pos(pos: str) -> int:
    """
    convert pos like: DET -> 3 (because POS_CLASSES[3] = 'DET')
    """
    return POS_CLASSES.index(pos)


def get_convert_function(task: str) -> Callable[[str], int]:
    if task == 'get_pos':
            return convert_pos
    if task == 'get_morphy':
        return convert_morphy