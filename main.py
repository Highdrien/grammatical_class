import os
import yaml
import argparse
from easydict import EasyDict
from icecream import ic
from typing import Optional

from src.train import train


def load_config(path: Optional[str]='config/config.yaml') -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def find_config(experiment_path: str) -> str:
    yaml_in_path = list(filter(lambda x: x[-5:] == '.yaml', os.listdir(experiment_path)))

    if len(yaml_in_path) == 1:
        return os.path.join(experiment_path, yaml_in_path[0])

    if len(yaml_in_path) == 0:
        print("ERROR: config.yaml wasn't found in", experiment_path)
    
    if len(yaml_in_path) > 0:
        print("ERROR: a lot a .yaml was found in", experiment_path)
    
    exit()

IMPLEMENTED = ['train']

def main(options: dict) -> None:

    assert options['mode'] in IMPLEMENTED, f"Error, expected mode must in {IMPLEMENTED} but found {options['mode']}"

    if options['mode'] == 'train':
        config = load_config(options['config_path'])
        ic(config)
        train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'data'")
    parser.add_argument('--config_path', default=os.path.join('config', 'config.yaml'), type=str, help="path to config (for training)")
    parser.add_argument('--path', type=str, help="experiment path (for test, prediction or generate)")

    args = parser.parse_args()
    options = vars(args)

    main(options)