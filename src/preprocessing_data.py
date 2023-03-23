import os
import random
import numpy as np


def prepare_data(config, file_name):
    """
    Prepare the txt file found in config.data_path
    :param config: data config
    :param file_name: name of the txt file
    :return: sentences (list of sentences), labels (list of sentences labels)
             where a sentence is a list of words and its label is the list of labels for each of its words
    """
    sentences = []
    labels = []

    # Test if the file can be accessed
    path_file = os.path.join(config.data_path, file_name)
    try:
        fic = open(path_file, 'r', encoding='utf8')
    except IOError:
        print("The file: ", file_name, " was not found in  :", config.data_path)
        exit()

    pass_step = False
    # We go through the lines
    for line in fic:

        # Empty lines are ignored
        if line[0] == '\n':
            continue

        # We look for the beginning of the sentences
        if '# text =' in line:
            pass_step = False
            sentences.append([])
            labels.append([])
            continue

        # If we have a tab
        if line[0] == '#' or pass_step:
            continue

        # Si on a une tabulation
        tokenizedLine = line.split("\t")

        # ignore multi token words
        if '-' in tokenizedLine[0] or tokenizedLine[3] == '_' or tokenizedLine[1] == '_':
            continue

        # If we have a class 'X' we delete the whole represented list from the sentence
        if tokenizedLine[3] == 'X':
            pass_step = True
            del sentences[-1]
            del labels[-1]
            continue

        # On affecte tous les mots à la liste 
        sentences[-1].append(tokenizedLine[1])
        labels[-1].append(tokenizedLine[3])

    return sentences, labels


def create_word_dictionary(config, sentences):
    """
    Creates a dictionary from the words in X_dataset
    :param config: data config
    :param sentences: Sentence list (where sentences are lists of words)
    :return: dictionary with all words from X_dataset
    """
    index = 0
    dictionnaire = {}
    # On parcourt tous les mots de toutes les phrases
    for sentence in sentences:
        for word in sentence:
            if word.lower() not in dictionnaire:  # Si le mot n'est pas dans le dictionnaire, on le rajoute
                dictionnaire[word.lower()] = index
                index += 1

    dictionnaire[config.pad_character.lower()] = index  # On rajoute le pad_character dans le dictionnaire
    return dictionnaire


def create_label_dictionary(config, label):
    """
    Creates a list containing all labels
    :param config: data config
    :param label: List of all labels of the sentences
    :return: List of all possible labels of the words
    """
    label_dictionary = []
    for label_sentence in label:
        for label_word in label_sentence:
            # Si le label n'est pas dans la liste de tous les labels, on le rajoute
            if label_word not in label_dictionary:
                label_dictionary.append(label_word)

    label_dictionary.append(config.pad_character)  # On ajoute le label qui correspond au pad_caractère
    return label_dictionary


def dummy_sequence(config, sentences, label):
    """
    splits a list into sublists of equal length and fills the last sublists with a fill value if necessary to reach the
    specified sequence length. The splitting has no overlap.
    :param config: data config
    :param sentences: List of sentences
    :param label: List of labels of sentences
    :return: X_seq (List of sentences of the same size), Y_seq (List of labels of sentences of the same size)
    """
    x_sequences = []
    y_sequences = []

    seq_size = config.sequence_size  # Longueur des sequences
    pad = config.pad_character  # Caractère de complétion

    for i, x in enumerate(sentences):
        k = len(x) // seq_size  # k est le nombre de fois qu'on peut rentrer une sequence dans la phrase

        # On fait déjà les k première sequences
        for j in range(k):
            x_sequences.append(x[j * seq_size: (j + 1) * seq_size])
            y_sequences.append(label[i][j * seq_size: (j + 1) * seq_size])

        # Si on doit completer, prend la fin de la phrase et on l'a complete avec le pad_character
        if len(x) != k:
            x_sequences.append(x[k * seq_size:] + [pad] * ((k + 1) * seq_size - len(x)))
            y_sequences.append(label[i][k * seq_size:] + [pad] * ((k + 1) * seq_size - len(x)))

    return x_sequences, y_sequences


def smart_sequence(config, sentences):
    """
    Splits a list into sublists of equal length and fills the last sublists with a fill value if necessary to reach the
    specified sequence length. The splitting is done in such a way that the sub-lists overlap rather than being disjoint
    :param config: data config
    :param sentences: List of sentences
    :return: X_seq (List of sentences of the same size)
    """
    sequences = []

    seq_size = config.sequence_size  # Longueur des sequences
    pad = config.pad_character  # Caractère de complétion

    for i, sentence in enumerate(sentences):
        a, b = 0, seq_size
        sentence += [pad] * seq_size
        while sentence[a] != pad:
            sequences.append(sentence[a:b])
            a += seq_size // 2
            b += seq_size // 2

    return sequences


def word_to_number(config, x_sequences, y_sequences, word_dictionary, label_dictionary):
    """
    Replaces all words with their dictionary indices and all labels with their indices in the list of labels
    :param config: data config
    :param x_sequences: str matrix, representing the sequences
    :param y_sequences: str matrix, representing the labels of the words in each sequence
    :param word_dictionary: the word dictionary
    :param label_dictionary: the list of all labels
    :return: X and Y int matrices, representing sequences and labels (in hot-one encoding)
    """
    X, Y = [], []
    nb_mots = len(word_dictionary)

    for i in range(len(x_sequences)):
        x, y = [], []
        for j in range(config.sequence_size):

            # Si le mot est dans le dico, on met son indice.
            # Sinon, on met le nombre de mots du dictionnaire (qui est l'indice maximal +1)
            if x_sequences[i][j].lower() in word_dictionary and random.random() > config.rate_not_add_to_dictionary:
                x.append(word_dictionary[x_sequences[i][j].lower()])
            else:
                x.append(nb_mots)

            # Si le label est dans la liste des labels, alors on met par son indice (en hot-one encoding).
            # Sinon, on met [0, 0, 0, ..., 0, 1]
            if y_sequences[i][j] in label_dictionary:
                hot_one = [0] * len(label_dictionary)
                hot_one[label_dictionary.index(y_sequences[i][j])] = 1
                y.append(hot_one)
            else:
                y.append([0] * (len(label_dictionary) - 1) + [1])

        X.append(x)
        Y.append(y)
    return X, Y


def get_word_and_label_dictionary(config, file_name):
    """
    Get the word and label dictionary
    :param config: main config
    :param file_name: train file
    """
    sentences, label = prepare_data(config.data, file_name)

    # split to have train and validation data
    split = int(config.train.split * len(sentences))
    train_sentences = sentences[:split]
    train_label = label[:split]

    # create word and label dictionary
    word_dictionary = create_word_dictionary(config.data, train_sentences)
    label_dictionary = create_label_dictionary(config.data, train_label)

    return word_dictionary, label_dictionary