import torch
import numpy as np
from icecream import ic
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import accuracy_score

# def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> np.ndarray: 
#     """
#     On renvoie un numpy array avec les valeurs de toutes les metrics: Accuracy micro, mean Accuracy macro
#     """
#     metrics_value = []
#     argmax_pred = torch.argmax(y_pred, dim=1)
#     metrics_value += accuracy_without_pad(argmax_pred, y_true, num_classes)
#     return np.array(metrics_value)


def accuracy_without_pad(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> list:
    """
    y_pred : [B, C, K] (C = num_classes, K = max_len)
    y_true : [B, K]
    On récupère la micro et la moyenne du macro
    """
    pad_index = 10 # A ne pas considérer du y_true
    mask = (y_true != pad_index ) 
    masked_y_pred = y_pred[mask]
    masked_y_true = y_true[mask]
    accuracy_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
    accuracy_macro.update(masked_y_pred, masked_y_true)
    accuracy_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")
    accuracy_micro.update(masked_y_pred, masked_y_true)
    macro = torch.mean(accuracy_macro.compute()).item()
    micro = accuracy_micro.compute().item()
    return [macro, micro]




def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> float:
    """
    Calcule l'accuracy entre y_true et y_pred sans considérer le padding.
    y_pred : [B, L, C] (B = batch_size,L = sequence_lenght, C = num_classes)
    y_true : [B, L, C] (B = batch_size,L = sequence_lenght, C = num_classes)
    """
    
    # Convert probabilities to class predictions
    y_pred_classes = torch.argmax(y_pred, dim=2)
    
    # Convert one-hot encoded labels to class indices
    y_true_classes = torch.argmax(y_true, dim=2)
    
    # Compute accuracy
    correct_predictions = torch.eq(y_pred_classes, y_true_classes)
    accuracy_value = correct_predictions.sum().item() / correct_predictions.numel()

    return accuracy_value


def create_batch(batch_size, sequence_length, num_classes):
    """
    Create a batch of one-hot vectors as PyTorch tensors.

    Parameters:
    - batch_size: Size of the batch.
    - sequence_length: Length of each sequence.
    - num_classes: Number of classes.

    Returns:
    - A PyTorch tensor representing the batch.
    """
    # Initialize an empty tensor for the batch
    batch = torch.zeros((batch_size, sequence_length, num_classes), dtype=torch.float32)

    # Loop through each sequence in the batch
    for i in range(batch_size):
        # Randomly select indices to set to 1 in each sequence
        indices = torch.randint(0, num_classes, (sequence_length,))
        batch[i, torch.arange(sequence_length), indices] = 1.0

    return batch

# if __name__ == "__main__":
#     y_pred = torch.rand((10, 19, 10))
#     y_true = torch.rand((10, 19, 10))
#     ic(compute_metrics(y_pred, y_true, 19))
#     print(compute_metrics(y_pred, y_true, 19))


if __name__ == "__main__": 
    #y_pred de shape [B,L,C] (B=batch_size, L=sequence_lenght, C=num_classes)
    #y_true de shape [B,L,C] (B=batch_size, L=sequence_lenght, C=num_classes)

    #create a batch of size 10 containing 12 random one hot vectors of size 19
    y_true=create_batch(1000,12,19)
    y_pred=create_batch(1000,12,19)
    
    
    
    print("Shape of y_pred:", y_pred.shape)
    print("Shape of y_true:", y_true.shape)
    print(compute_metrics(y_pred, y_true, 19))