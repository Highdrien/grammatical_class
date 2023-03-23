import os
import yaml
import argparse
from easydict import EasyDict as edict

from src.train import train
from src.test import test
from src.predict import prediction


def load_config(path='configs/config.yaml'):
    stream = open(path, 'r')
    return edict(yaml.safe_load(stream))


def __train(config_path):
    config = load_config(path=config_path)
    train(config)


def __test(experiment_path):
    config = load_config(path=os.path.join(experiment_path, 'config.yaml'))
    test(experiment_path, config)


def __prediction(experiment_path):
    config = load_config(path=os.path.join(experiment_path, 'config.yaml'))
    prediction(experiment_path, config)


def main(options):
    """
    launches a training or a test or a prediction depending on the chosen mode.

    To run a training, set --mode train and --config <path_to_your_config>.
        By default, the path_to_your_config = 'configs/configs.yaml'.
        The training will run, and you can find the results in the log folder.

    To perform a test or a prediction, put --mode 'test' or 'predict' and -- experiment_path <your_experiment_path>
        which is a folder containing: 'config.yaml' and the model weights (.h5 file)
    """
    if options['mode'] == 'train':
        __train(options['config_path'])

    elif options['mode'] == 'test':
        __test(options['experiment_path'])

    elif options['mode'] == 'predict':
        __prediction(options['experiment_path'])

    else:
        raise "choose a mode between 'train', 'test' and 'predict'"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument('--mode', default=None, type=str, help="choose a mode between 'train', 'test' and 'predict'")
    parser.add_argument('--config_path', default='configs\\config.yaml', type=str, help="path to config (just for training)")
    parser.add_argument('--experiment_path', type=str, help="experiment path")

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict

    main(options)
