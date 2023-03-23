import os
import numpy as np

from src.preprocessing_data import get_word_and_label_dictionary
from src.model import create_model

LABEL = {'ADJ': 'adjectif', 'ADP': 'préposition', 'ADV': 'adverbe', 'AUX': 'auxiliaire',
         'CCONJ': 'conjonction de coordination', 'DET': 'déterminant', 'INTJ': 'interjection', 'NOUN': 'nom',
         'NUM': 'nombre', 'PART': 'particule', 'PRON': 'pronom', 'PROPN': 'nom propre', 'PUNCT': 'ponctuation',
         'SCONJ': 'conjonction de subordination', 'SYM': 'symbole', 'VERB': 'verbe'}


def get_sentences(config):
    """
    Get a list of each word from the texte file to predict
    :param config: predict config
    :return: list of words
    """
    data_name_to_predict = config.file_to_predict

    data_path = os.path.join(config.path, data_name_to_predict)
    sentence = []

    # Open the text file
    try:
        fic = open(data_path, 'r', encoding='utf8')
    except IOError:
        print("le fichier :", data_name_to_predict, "n'a pas été trouvé dans le chemin :", config.path)
        exit()

    # We get word in sentence, but we still have to separate the words from the punctuation
    for line in fic:
        line_lower = line.lower()
        sentence += line_lower.split(" ")

    # Open the punctuation file
    try:
        punctuation_file = open(os.path.join(config.path, config.punctuation_file), 'r', encoding='utf8')

    except IOError:
        print("the punctuation file was not found")
        exit()

    punctuation = []
    for line in punctuation_file:
        punctuation += line.split(" ")

    new_sentence = []
    for word in sentence:
        new_sentence += split_punctuation(word, punctuation)

    return new_sentence


def split_punctuation(word, punctuation):
    """
    Separates punctuation from words
    """
    forbidden_character = ('\n', ' ')

    new_words = []
    indice = 0
    for idx, character in enumerate(word):
        if character in forbidden_character:
            if indice != idx:
                new_words.append(word[indice:idx])
            indice = idx + 1
        elif character == "'" and idx != 0:
            new_words.append(word[indice:idx + 1])
            indice = idx + 1
        elif character in punctuation:
            if indice != idx:
                new_words.append(word[indice:idx])
            new_words.append(character)
            indice = idx + 1

    if indice != len(word) and word[indice:] not in forbidden_character:
        new_words.append(word[indice:])

    return new_words


def word_to_number(config, x_sequence, word_dictionary):
    """
    Replaces all words with their dictionary indices and all labels with their indices in the list of labels
    :param config: data config
    :param x_sequence: str matrix, representing sequences
    :param word_dictionary: the word dictionary
    :return: X an int matrix, representing the sequences
    """
    X = []
    nb_mots = len(word_dictionary)
    for i in range(len(x_sequence)):
        x = []
        for j in range(config.sequence_size):
            if x_sequence[i][j] in word_dictionary:
                x.append(word_dictionary[x_sequence[i][j]])
            else:
                x.append(nb_mots)
        X.append(x)
    return X


def dummy_sequence(config, sentences):
    """
    splits a list into sublists of equal length and fills the last sublists with a fill value if necessary to reach the
    specified sequence length. The splitting has no overlap.
    :param config: data config
    :param sentences: List of sentences
    :param label: List of labels of sentences
    :return: X_seq (List of sentences of the same size), Y_seq (List of labels of sentences of the same size)
    """
    x_sequence = []
    seq_size = config.sequence_size  # Longueur des sequences
    pad = config.pad_character  # Caractère de complétion
    for i, x in enumerate(sentences):
        k = len(x) // seq_size
        for j in range(k):
            x_sequence.append(x[j * seq_size: (j + 1) * seq_size])
        if len(x) != k:
            x_sequence.append(x[k * seq_size:] + [pad] * ((k + 1) * seq_size - len(x)))
    return x_sequence


def smart_sequence(config, sentence):
    """
    Splits a list into sublists of equal length and fills the last sublists with a fill value if necessary to reach the
    specified sequence length. The splitting is done in such a way that the sub-lists overlap rather than being disjoint
    :param config: data config
    :param sentence: one sentence
    :return: X_seq (List of sentences of the same size), Y_seq (List of labels of sentences of the same size)
    """
    sequences = []

    seq_size = config.sequence_size  # Longueur des sequences
    pad = config.pad_character  # Caractère de complétion

    a, b = 0, seq_size
    sentence += [pad] * seq_size
    while sentence[a] != pad:
        sequences.append(sentence[a:b])
        a += seq_size // 2
        b += seq_size // 2

    return sequences


def indice(config, sentence):
    """
    returns a list of the same shape as sentence, which for each i, result[i] will be the
    index list where we can find the word sentence[i] in smart_sequence(sentence)

    for example when we make a smart sequence with a sequence length of 20,
    we will find the 16th word of the sentence:
        - in 16 place in the 1st sequence
        - 6th place in the 2nd sequence
    so result[16] = [[0, 15], [1, 5]]
    """
    idx = []
    count = 0
    for _ in sentence:
        idx.append(count)
        count += 1

    idx = smart_sequence(config, idx)

    count = 0
    result = []
    begin = 0
    for _ in sentence:
        c, begin = find(count, idx, max(begin - 2, 0))
        result.append(c)
        count += 1

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


def prediction(experiment_path, config):
    """
    Predicts the grammatical classes of each word in the text file in another text file
    :param experiment_path: path to an experiment which contain a config and a model's weight
    :param config: main confing
    """
    # get the checkpoint_file
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

    # get data to be predicted and transform it into sequences
    train_data_name = config.language + '_train_data.txt'
    word_dictionary, label_dictionary = get_word_and_label_dictionary(config, train_data_name)
    sentence = get_sentences(config.predict)
    if config.data.create_sequence == 'dummy':
        x = word_to_number(config.data, dummy_sequence(config.data, [sentence]), word_dictionary)
    elif config.data.create_sequence == 'smart':
        x = word_to_number(config.data, smart_sequence(config.data, sentence), word_dictionary)
    else:
        raise "choose how to create sequence between 'dummy' or 'smart'"

    model = create_model(config, len(word_dictionary), len(label_dictionary))
    model.load_weights(checkpoint_file)

    y_pred = model.predict(x)
    if config.data.create_sequence == 'dummy':
        y_pred = np.argmax(y_pred, axis=2)
        y_pred = y_pred.reshape(np.shape(y_pred)[0] * np.shape(y_pred)[1])
        write_prediction(config, sentence, y_pred, label_dictionary)

    else:
        y_pred = merge(config.data, sentence, y_pred)
        write_prediction(config, sentence[:-config.data.sequence_size], y_pred, label_dictionary)


def write_prediction(config, sentence, y_pred, label_dictionary):
    """
   Writes the prediction to a new text file
   :param config: main config
   :param sentence: sentence
   :param y_pred: prediction of the model
   :param label_dictionary: dictionary of the labels
   """
    predict_file = config.predict.predict_file
    if predict_file is None:
        predict_file = config.predict.file_to_predict[:-len('.txt')] + '_predicted' + config.predict.file_to_predict[
                                                                                      -len('.txt'):]

    LABEL[config.data.pad_character] = 'pad_character'

    with open(os.path.join(config.predict.path, predict_file), 'w') as f:
        for i in range(len(sentence)):
            f.write(sentence[i] + ' : ' + LABEL[label_dictionary[y_pred[i]]] + '\n')


def merge(config, sentence, y_pred):
    """
    This function takes the original sentence and the prediction of the model from the smart_sequence(sentence) and
    returns result which has the same shape as sentence and contains the predicted label. For words that have multiple
    occurrences because of the smart sequence, takes the label that was predicted with the highest probability

    :param config: data config
    :param sentence: one sentence (list of word)
    :param y_pred: result of model.predict(x_sequence)
    """
    sentence = sentence[:-config.sequence_size]
    idx = indice(config, sentence)
    result = []
    y_argmax = np.argmax(y_pred, axis=2)
    for i in range(len(sentence)):
        if len(idx[i]) == 1:
            a, b = idx[i][0]
            result.append(y_argmax[a, b])
        else:
            a0, b0 = idx[i][0]
            a1, b1 = idx[i][1]
            if y_pred[a0, b0, y_argmax[a0, b0]] > y_pred[a1, b1, y_argmax[a1, b1]]:
                result.append(y_argmax[a0, b0])
            else:
                result.append(y_argmax[a1, b1])
    return result
