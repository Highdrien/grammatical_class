import torch
import torch.nn as nn

from icecream import ic


class MultiLabelLSTM(nn.Module):
    def __init__(self, embedding_dim, lstm_1, lstm_2, max_classes):
        super(MultiLabelLSTM, self).__init__()

        # Première partie du modèle - Prédiction du nombre de classes
        self.embedding = nn.Embedding(max_classes, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, lstm_1, batch_first=True)
        self.num_classes_output = nn.Linear(lstm_1, 1)

        # Deuxième partie du modèle - Prédiction des classes
        self.lstm2 = nn.LSTM(embedding_dim + lstm_1, lstm_2, batch_first=True)
        self.classes_output = nn.Linear(lstm_2, max_classes)

    def forward(self, input_sequence, num_classes_input):
        # Première partie du modèle
        embedded = self.embedding(input_sequence)
        ic(embedded.shape)
        lstm_out1, _ = self.lstm1(embedded)
        ic(lstm_out1.shape)
        num_classes_pred = self.num_classes_output(lstm_out1[:, -1, :])
        ic(num_classes_pred.shape)

        # Deuxième partie du modèle
        num_classes_input = num_classes_input.view(-1, 1, 1).expand(-1, 1, lstm_out1.size(2))
        ic(num_classes_input.shape)
        repeated_num_classes = num_classes_input.repeat(1, embedded.size(1), 1)
        ic(repeated_num_classes.shape)
        lstm2_input = torch.cat([embedded, repeated_num_classes], dim=-1)
        ic(lstm2_input.shape)
        lstm_out2, _ = self.lstm2(lstm2_input)
        ic(lstm_out2.shape)
        classes_pred = self.classes_output(lstm_out2)
        ic(classes_pred.shape)

        return num_classes_pred, classes_pred
    


if __name__ == '__main__':

    B = 10      # batch size
    K = 20      # sequence length
    C = 10      # num classes
    E = 32      # embedding dim
    L1 = 32     # lstm dim
    L2 = 64     # lstm dim 2


    X = torch.randint(0, C, size=(B, K))
    y1 = torch.randint(1, 11, size=(B, 1)) # Nombre de classes
    y2 = torch.randint(0, C, size=(B, K))  # Classes correspondantes

    ic(X.shape)
    ic(y1.shape)
    ic(y2.shape)

    model = MultiLabelLSTM(embedding_dim=E, lstm_1=32, lstm_2=64, max_classes=C)

    num_classes_pred, classes_pred = model(X, y1)

    

    