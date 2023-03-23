import os
import numpy as np

from src.preprocessing_data import prepare_data, dummy_sequence, smart_sequence, word_to_number, \
    get_word_and_label_dictionary
from src.model import create_model
from configs.utils import test_logger


def test(experiment_path, config):
    """
    makes a test of an already trained model and creates a test_log.csv file in the experiment_path containing the
    metrics values at the end of the test

    :param experiment_path: path of the experiment folder, containing the config, and the model weights in an .h5 file
    :config: configuration of the model
    """
    index = -1
    for i, file in enumerate(os.listdir(experiment_path)):
        if len(file) > 3 and '.h5' == file[-3:]:
            if index == -1:
                index = i
            else:
                print("Warning: found more than one model's weight in", experiment_path)
    if index == -1:
        raise "No model's weight was found in "

    checkpoint_file = os.path.join(experiment_path, os.listdir(experiment_path)[index])

    test_data_name = config.language + '_test_data.txt'
    train_data_name = config.language + '_train_data.txt'

    word_dictionary, label_dictionary = get_word_and_label_dictionary(config, train_data_name)

    test_sentences, test_label = prepare_data(config.data, test_data_name)

    if config.data.create_sequence == 'dummy':
        x_sequences, y_sequences = dummy_sequence(config.data, test_sentences, test_label)
    elif config.data.create_sequence == 'smart':
        test_sentences_copy = [[test_sentences[i][j] for j in range(len(test_sentences[i]))] for i in range(len(test_sentences))]
        test_label_copy = [[test_label[i][j] for j in range(len(test_label[i]))] for i in range(len(test_label))]
        x_sequences = smart_sequence(config.data, test_sentences_copy)
        y_sequences = smart_sequence(config.data, test_label_copy)
    else:
        raise "Choose how to create sequence between 'dummy' or 'smart'"

    X_test, Y_test = word_to_number(config.data, x_sequences, y_sequences, word_dictionary, label_dictionary)

    model = create_model(config, len(word_dictionary), len(label_dictionary))

    model.load_weights(checkpoint_file)
    metrics = ['loss'] + list(config.metrics.keys())

    values = model.evaluate(X_test, Y_test, batch_size=config.test.batch_size)

    if config.data.create_sequence == 'smart':
        y_pred = model.predict(X_test, batch_size=config.test.batch_size)
        y_pred = merge(config.data, test_sentences, y_pred)
        y_true = label_to_number(test_label, label_dictionary)
        acc = accuracy(y_true, y_pred)
        metrics.append('accuracy_with_merge')
        values.append(acc)

    test_logger(experiment_path, metrics, values)


def indice(config, sentences):
    """
    returns a list of the same shape as sentences, which for each i, result[i] will be the
    index list where we can find the word sentences[i] in smart_sequence(sentences)

    for example when we make a smart sequence with a sequences length of 20,
    we will find the 16th word of the 1st sentence:
        - in 16 place in the 1st sequence
        - 6th place in the 2nd sequence
    so result[0][16] = [[0, 15], [1, 5]]
    """
    idx = []
    count = 0
    for sentence in sentences:
        tmp = []
        for _ in sentence:
            tmp.append(count)
            count += 1
        idx.append(tmp)

    idx = smart_sequence(config, idx)

    count = 0
    result = []
    begin = 0
    for sentence in sentences:
        tmp = []
        for _ in sentence:
            c, begin = find(count, idx, max(begin - 2, 0))
            tmp.append(c)
            count += 1
        result.append(tmp)

    return result


def find(x, m, begin=0):
    """
    find the element x in a matrix m knowing that:
    - x necessarily appears once or twice in m
    - x necessarily appears after the "begin" line
    - if x appears twice, then the 2 appearances of x in m are necessarily one line apart

    :param x: element of m
    :param m: matrix which contain x
    :param begin: index of a line of m such that m[:begin] does not contain x
    :return: list of indices of the occurrence of x and the last line of the occurrence of x
    """
    idx = 0
    c = []
    for i in range(begin, len(m)):
        for j in range(len(m[i])):
            if m[i][j] == x:
                c.append([i, j])
                idx = i
                # c can't be bigger than 2
                if len(c) == 2:
                    return c, idx
        if i > idx + 2:
            return c
    return c, idx


def merge(config, sentences, y_pred):
    """
    This function takes original sentences and the prediction of the model from the smart_sequence(sentences) and
    returns result which has the same shape as sentences and contains the predicted label. For words that have multiple
    occurrences because of the smart sequence, takes the label that was predicted with the highest probability

    :param config: data config
    :param sentences: list of sentences
    :param y_pred: result of model.predict(x_sequence)
    """
    idx = indice(config, sentences)
    result = []
    y_argmax = np.argmax(y_pred, axis=2)
    for i in range(len(sentences)):
        tmp = []
        for j in range(len(sentences[i])):
            if len(idx[i][j]) == 1:
                a, b = idx[i][j][0]
                tmp.append(y_argmax[a, b])
            else:
                a0, b0 = idx[i][j][0]
                a1, b1 = idx[i][j][1]
                # print(a0, b0, y_argmax[a0, b0], a1, b1, y_argmax[a0, b0])
                if y_pred[a0, b0, y_argmax[a0, b0]] > y_pred[a1, b1, y_argmax[a1, b1]]:
                    tmp.append(y_argmax[a0, b0])
                else:
                    tmp.append(y_argmax[a1, b1])
        result.append(tmp)
    return result


def label_to_number(y_sentences, label_dictionary):
    """
    replaces the elements of y_sentences with their values in the label_dictionary
    """
    Y = []
    for i in range(len(y_sentences)):
        y = []
        for j in range(len(y_sentences[i])):
            if y_sentences[i][j] in label_dictionary:
                y.append(label_dictionary.index(y_sentences[i][j]))
            else:
                y.append(len(label_dictionary))
        Y.append(y)
    return Y


def accuracy(y_true, y_pred):
    """
    calculates the accuracy
    y_true and y_pred have the same shape but the size of the lines are not the same
    """
    correct = 0
    elements = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            elements += 1
            if y_true[i][j] == y_pred[i][j]:
                correct += 1
    return 100 * correct / elements