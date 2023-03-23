import os
from datetime import datetime
from easydict import EasyDict as edict


def number_folder(path, name):
    """
    finds a declination of a folder name so that the name is not already taken
    """
    elements = os.listdir(path)
    last_index = -1
    for i in range(len(elements)):
        folder_name = name + str(i)
        if folder_name in elements:
            last_index = i
    return name + str(last_index + 1)


def train_logger(config, computed_metrics):
    """
    creates a logs folder where we can find the config in confing.yaml and
    the values of the loss and metrics according to the epochs in train_log.csv
    """
    path = config.train.logs_path
    folder_name = number_folder(path, 'experiment_')
    path = os.path.join(path, folder_name)
    os.mkdir(path)
    print(f'{path = }')

    # create train_log.csv where save the metrics
    with open(os.path.join(path, 'train_log.csv'), 'w') as f:
        first_line = 'step,'
        for key in computed_metrics.keys():
            first_line += key + ','
        f.write(first_line[:-1] + '\n')

        for i in range(len(computed_metrics['loss'])):
            line = str(i) + ','
            for key in computed_metrics.keys():
                line += str(computed_metrics[key][i]) + ','
            f.write(line[:-1] + '\n')

    # copy the config
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        f.write("config_metadata: 'Saving time : " + date_time + "'\n")
        for line in config_to_yaml(config):
            f.write(line + '\n')

    return path


def config_to_yaml(config, space=''):
    """
    transforms a dictionary (config) into a yaml line sequence
    """
    config_str = []
    for key, value in config.items():
        if type(value) == edict:
            config_str.append('')
            config_str.append('# ' + key + ' options')
            config_str.append(key + ':')
            config_str += config_to_yaml(value, space='  ')
        elif type(value) == str:
            config_str.append(space + key + ": '" + str(value) + "'")
        elif value is None:
            config_str.append(space + key + ": null")
        elif type(value) == bool:
            config_str.append(space + key + ": " + str(value).lower())
        else:
            config_str.append(space + key + ": " + str(value))
    return config_str


def test_logger(path, metrics, values):
    with open(os.path.join(path, 'test_log.csv'), 'w') as f:
        for i in range(len(metrics)):
            f.write(metrics[i] + ': ' + str(values[i]) + '\n')