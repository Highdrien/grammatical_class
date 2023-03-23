import os
import matplotlib.pyplot as plt

from src.preprocessing_data import prepare_data, create_word_dictionary, create_label_dictionary, dummy_sequence, \
    smart_sequence, word_to_number
from src.model import create_model
from configs.utils import train_logger


def train(config):
    """ train the model
    :param config: main config
    """
    # Get data
    file_name = config.language + '_train_data.txt'
    sentences, label = prepare_data(config.data, file_name)

    # split to have train and validation data
    split = int(config.train.split * len(sentences))

    train_sentences, val_sentences = sentences[:split], sentences[split:]
    train_label, val_label = label[:split], label[split:]

    # create word and label dictionary
    word_dictionary = create_word_dictionary(config.data, train_sentences)
    label_dictionary = create_label_dictionary(config.data, train_label)

    # Create sequences
    if config.data.create_sequence == 'dummy':
        x_train_sequences, y_train_sequences = dummy_sequence(config.data, train_sentences, train_label)
        x_val_sequence, y_val_sequence = dummy_sequence(config.data, val_sentences, val_label)
    elif config.data.create_sequence == 'smart':
        x_train_sequences = smart_sequence(config.data, train_sentences)
        y_train_sequences = smart_sequence(config.data, train_label)
        x_val_sequence, y_val_sequence = smart_sequence(config.data, val_sentences, val_label)
    else:
        raise "Choose how to create sequence between 'dummy' or 'smart'"

    # Formatting train data
    X_train, Y_train = word_to_number(config.data, x_train_sequences, y_train_sequences, word_dictionary,
                                      label_dictionary)

    # Formatting validation data
    X_val, Y_val = word_to_number(config.data, x_val_sequence, y_val_sequence, word_dictionary, label_dictionary)

    model = create_model(config, len(word_dictionary), len(label_dictionary))

    history = model.fit(X_train, Y_train,
                        batch_size=config.train.batch_size,
                        epochs=config.train.epochs,
                        callbacks=None,
                        validation_data=(X_val, Y_val),
                        shuffle=True,
                        validation_batch_size=config.val.batch_size)

    path = train_logger(config, history.history)

    if config.train.save_learning_curves:
        save_learning_curves(history, os.path.join(path, 'learning_curves.png'))

    if config.train.save_checkpoint:
        checkpoint_path = os.path.join(path, config.model.type + '.h5')
        model.save_weights(checkpoint_path)
        print('checkpoint has been save in ', checkpoint_path)


def save_learning_curves(history, path):
    """
    save the accuracy and val_accuracy on each epoch to a png in path
    """
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(path)