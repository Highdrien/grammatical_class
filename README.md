# Gramatical classification

The goal of this repository is to find the grammatical class of each word from a text file containing sentences. For example, the prediction of the sentence "I think, therefore I am" will be:

| Word    | Class Word  |
| :---:   |:-------:    |
| I       |`pronoun`    |
|think    |`verb`       |
| ,       |`punctuation`|
|therefore|`abverb`     |
|I        |`pronoun`    |
|am       |`auxiliary`  |
|.        |`punctuation`|

You can find all the different grammatical classes of words on the website [here](https://universaldependencies.org/u/pos/index.html)

# Contents

- [Installation Requirements](#installation-requirements)
- [Configs](#configs)
- [Run the code](#run-the-code)
  - [Train](#train)
  - [Test](#test)
  - [Prediction](#prediction)
- [trained model](#trained-model)
  - [experiment\_0: dummy sequences](#experiment_0-dummy-sequences)
  - [experiment\_1: smart sequences](#experiment_1-smart-sequences)


# Installation Requirements

To run the code you need python 3.9.13 and packages in the following versions :

- keras==2.11.0
- matplotlib==3.6.2
- numpy==1.23.0
- easydict==1.10
- PyYAML==6.0

You can run the following code to install all packages in the correct versions:
```bash
pip install -r requirements.txt
```

# Configs

If you want to start a training session, you must first make a configuration. A basic configuration is available in `configs/config.yaml`.

You will be able to choose many parameters, the most important of which are:
- `language`: the language you want your AI to train in. You can choose between 'fr' for French, 'en' for English and 'de' for German
- `model.type`: you can choose the type of model between an LSTM or a GRU

As the sentences are not of the same length, they have to be cut into sequences. You can modify the sequence with the following parameters:
- `data.sequence_size`: the length of the sequences. 
- `data.create_sequence`: the way to cut the sentences into sequences. There are 2 possibilities: 
    - 'dummy': This is the most basic way. We cut the sentences without overlapping and add a completion character at the end to get the right size. For example, the sentence:\
   ['`I`', '`think`', '`,`', '`threfore`', '`I`', '`am`', '`.`']\
    will become\
   [['`I`', '`think`', '`,`', '`threfore`'], ['`I`', '`am`', '`.`', '`PAD`']]\
   if sequence_size is 4 and the completion character is '`PAD`'. 
    - 'smart': In this way, sentences are cut with an overlap. For example, the sentence:\
   ['`I`', '`think`', '`,`', '`threfore`', '`I`', '`am`', '`.`']\
    will become (with sequence size = 4)\
   [['`I`', '`think`', '`,`', '`threfore`'], ['`,`', '`threfore`', '`I`', '`am`'], ['`I`', '`am`', '`.`', '`PAD`']]

You can also define the most basic things in a training like the `train.batch_size`, the number of `train.epochs` and if you want to save the model weights (`train.save_checkpoint`).

For more information, I invite you to look at the configs/configs.yaml file where everything is explained.

# Run the code

This code contains three main functions: train(), test() and prediction()

To run this script, you can run the script with the following options:

- `mode`: specifies the operating mode, which should be 'train', 'test' or 'predict'

- `config_path`: specifies the path to the configuration file (config.yaml), which is used only for the training mode ('train').\
  By default, the path_to_your_config = 'configs/configs.yaml'.\
The training will run, and you can find the results in the log folder.

- `path`: specifies the path to the experiment ('config.yaml' and the model weights (.h5 file)) used for the 'test' and 'predict' modes\
    this is done in such a way that if you train a model with a certain config that is going to be stored in
    'logs/experiment_1', to test or predict it, you just have to put --path logs/experiment_1

## Train

You can therefore run the script for a training session with the following command:
```bash
python main.py --mode 'train' --config_path <path to your config>
```
Once the training is finished, you can find the information in the log/experiment folder. You will find a copy of your configuration, a train_log.csv file with the loss and metrics values for each epoch. And if you have configured it, you will also find the model weights and the learning curve

## Test

And to run the script for a test or prediction with the following command:
```bash
python main.py --mode 'test' --path <path to your experiment>
```

Once the test is finished, you can find the test results in the file test_log.txt saved in the path you specified. These results are the values of the loss and metrics. If you performed a smart_sequence, you will also find the accuracy_merge, which is the accuracy after merging the sequences. (When doing a smart sequence, some words are duplicated, so we take the prediction of the model with the highest probability)

## Prediction

And to run the script for a test or prediction with the following command:
```bash
python main.py --mode 'predict' --path <path to your experiment>
```

To make a prediction you need to specify in the config:
- `predict.path`: path to a folder containing all the following files
- `predict.file_to_predict`: txt file to predict
- `predict.punctuation_file`: the punctuation.txt file
- `predict.predicted_file`: the name of the response file, if set to null, the name will be the name of the file to predict followed by '_predicted'

# trained model

You will also find in this repository 2 experiments content their configurations, model weights and performances

## experiment_0: dummy sequences

This experiment was done using an LSTM, trained in French over 5 epochs, and cutting the sentences without overlap (`data.create_sequence` = 'dummy')

<p align="center"><img src=logs/experiment_0/learning_curves.png><p>

Afterafter a test, we find: 
- loss = 0.22075200080871582
- accuracy = 0.9369408488273621

## experiment_1: smart sequences

This experiment was done using an LSTM, trained in French over 5 epochs, and cutting the sentences with overlap (`data.create_sequence` = 'smart')

<p align="center"><img src=logs/experiment_1/learning_curves.png><p>

Afterafter a test, we find:
- loss = 0.21609660983085632
- accuracy = 0.9424740672111511
- accuracy_with_merge = 92.8240024744819