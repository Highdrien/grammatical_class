import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from LSTM import LSTMClassifier  # Importer le modèle LSTM


class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Créer un seul exemple et son label
data = [torch.tensor([0, 1, 1, 0])]  # Remplacer par votre exemple
labels = [torch.tensor([0,1,0,2])]  # Remplacer par votre label

# Créer le Dataset
dataset = SimpleDataset(data, labels)

def load_config(config_path="config.yaml"):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. Default is "config.yaml".

    Returns:
        dict: The loaded configuration.

    """
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config

def set_seed(seed):
    """
    Set the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains the given model using the provided data loader, criterion, optimizer, and number of epochs.

    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): The data loader for training data.
        criterion (loss function): The loss function used for training.
        optimizer (optimizer): The optimizer used for training.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        None
    """
    model.train()
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            print("output_shape", outputs.shape)
            print("labels_shape", labels.shape)

            #outputs = outputs.permute(0, 2, 1) #PAS SUR DE CETTE LIGNE
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def evaluate_model(model, test_loader, criterion):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        criterion: The loss function used for evaluation.

    Returns:
        tuple: A tuple containing the average loss and accuracy of the model on the test dataset.
    """

    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            total_correct += correct
            total_count += labels.numel()

    average_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_count

    return average_loss, accuracy

def main():
    """
    Main function for training and evaluating a LSTM classifier model.

    This function loads the configuration, sets up the model parameters, loads the data,
    instantiates the model, defines the loss function and optimizer, trains the model,
    evaluates the model, and saves the best model.

    Returns:
        None
    """

    # Charger les paramètres du modèle
    config = load_config()

    # Paramètres du modèle
    INPUT_SIZE = config["sentence_size"]
    EMBEDDING_SIZE = config["embedding_size"]
    HIDDEN_SIZE = config["hidden_size"]
    OUTPUT_SIZE = config["num_classes"]
    NUM_EPOCHS = config["num_epochs"]
    BATCH_SIZE = config["batch_size"]

    # Autres paramètres
    seed = config["seed"]
    model_save_path = config["model_save_path"]

    # Charger les données
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Instancier le modèle
    model = LSTMClassifier(INPUT_SIZE, EMBEDDING_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)

    # Définir la fonction de perte et l'optimiseur
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Fixer la seed pour la reproductibilité
    set_seed(seed)

    # Entraînement du modèle
    train_model(model, train_loader, criterion, optimizer, num_epochs=NUM_EPOCHS)

    # Évaluation du modèle
    average_loss, accuracy = evaluate_model(model, test_loader, criterion)

    print(f'Average loss: {average_loss}, Accuracy: {accuracy}')

    # Enregistrement du meilleur modèle
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    main()
