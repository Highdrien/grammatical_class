import os
import yaml
from easydict import EasyDict

from src.train import train
from config.process_config import process_config
import os


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def load_config(path: str) -> EasyDict:
    stream = open(path, 'r')
    return EasyDict(yaml.safe_load(stream))


def main() -> None:
    config = load_config(os.path.join('config', 'config.yaml'))
    process_config(config)
    train(config)


if __name__ == "__main__":
    main()