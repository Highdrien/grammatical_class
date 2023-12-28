import os
import json
from typing import List, Union


POS_CLASSES = ['PRON', 'AUX', 'VERB', 'DET', 'ADJ', 'NOUN', 'ADP', 'PROPN', 'NUM', 'CCONJ',
               '<PAD>', 'ADV', 'PART', 'INTJ', 'SYM', 'PUNCT', 'SCONJ', 'X', '_']


class Convert():
    def __init__(self) -> None:
        self.label = []
        self.num_classes = 0
    def convert(self, label_to_convert: str) -> Union[int, List[int]]:
        raise NotImplementedError
    def find_num_classes(self) -> int:
        return len(self.label)
    

class POS(Convert):
    def __init__(self) -> None:
        super().__init__()
        self.label = POS_CLASSES

    def convert(self, label_to_convert: str) -> int:
        return self.label.index(label_to_convert)


class Morphy(Convert):
    def __init__(self) -> None:
        super().__init__()
        morphy = os.path.join('src', 'dataloader', 'morphy.json')
        with open(file=morphy, mode='r', encoding='utf8') as f:
            self.label = json.load(f)
            f.close()
        self.num_classes = self.find_num_classes()
    
    def convert(self, label_to_convert: str) -> List[int]:
        label_to_convert = dict(map(lambda x: self.split_x(x=x), label_to_convert.split('|')))

        output = []
        for key, value in self.label.items():
            if key in label_to_convert:
                output.append(value.index(label_to_convert[key]))
            else:
                output.append(value.index('Not'))
        
        return output

    def split_x(self, x: str) -> List[str]:
        if x in ['_', '<PAD>']:
            return [x, 'Yes']
        return x.split('=')


def get_convert(task: str) -> Convert:
    if task == 'get_pos':
        convert = POS()
    if task == 'get_morphy':
        convert = Morphy()
    return convert
    

if __name__ == '__main__':
    from icecream import ic
    convert = get_convert(task='get_morphy')
    o1 = convert.convert(label_to_convert='Emph=No|Number=Sing|Person=1|PronType=Prs')
    o2 = convert.convert(label_to_convert='_')
    ic(o1)
    ic(o2)