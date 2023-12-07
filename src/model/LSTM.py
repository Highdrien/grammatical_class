import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd


# Définir le modèle LSTM

#Forme de l'entrée: (batch_size, sequence_length) on donne au réseau un vecteur d'identifiants de mots EX: [1, 3, 5, 4, 7]
#Forme de la sortie: (batch_size, sequence_length, num_classes) on veut que le réseau nous donne un vecteur de probabilités pour chaque mot de la séquence EX: [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # Utilisez dim=2 pour softmax sur la dimension temporelle

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out)
        output = self.softmax(output)
        return output

