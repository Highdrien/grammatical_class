import keras
from keras import layers


def create_model(config, nb_words, nb_labels):
    """
    Create a model LSTM or GRU (depends on the config)
    :param config: main config
    :param nb_words: number of words in the dictionary
    :param nb_labels: number of labels
    :return: model
    """

    model = keras.Sequential()
    model.add(layers.Embedding(nb_words + 1, config.model.embedding_dim, input_length=config.data.sequence_size))

    if config.model.type == 'LSTM':
        model.add(layers.Bidirectional(layers.LSTM(config.model.hidden_size, return_sequences=True)))
    elif config.model.type == 'GRU':
        model.add(layers.Bidirectional(layers.GRU(config.model.hidden_size, return_sequences=True)))
    else:
        print("Error: chose a mode witch 'LSTM' or 'GRU'")
        exit()

    model.add(layers.Dense(config.model.dense, activation='relu'))
    model.add(layers.Dropout(config.model.dropout))
    model.add(layers.Dense(nb_labels, activation='softmax'))

    if config.model.model_summary:
        model.summary()

    model.compile(loss=config.model.loss, optimizer=config.model.optimizer, metrics=config.metrics.values())
    return model