import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

from LSTM import LSTMClassifier # Importer le modèle LSTM




# Paramètres du modèle
input_size = vocab_size  # Taille du vocabulaire
embedding_size = 50  # Taille de l'embedding
hidden_size = 100  # Taille de la couche cachée LSTM
output_size = num_classes  # Nombre de classes de sortie
num_epochs = 10  # Nombre d'époques



# Instancier le modèle
model = LSTMClassifier(input_size, embedding_size, hidden_size, output_size)

# Définie la shape d'entrée et de sortie du modèle
#Shape d'entrée : (batch_size, seq_length)

# Définir la fonction de perte et l'optimiseur
criterion = nn.NLLLoss()   #négative log likelihood loss
optimizer = optim.Adam(model.parameters(), lr=0.001)


#définir le dataloader



# Entraînement du modèle (assurez-vous d'avoir vos données d'entraînement)
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 1)  # Permute pour correspondre à la forme attendue par NLLLoss
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Évaluation du modèle (assurez-vous d'avoir vos données de test)
model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        outputs = outputs.permute(0, 2, 1)  # Permute pour correspondre à la forme attendue par NLLLoss
        # Effectuez l'évaluation, par exemple, calculez la précision, la perte, etc.
