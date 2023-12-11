import torch
import numpy as np
from icecream import ic
from torcheval.metrics import MulticlassAccuracy


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> np.ndarray: 
    """
    On renvoie un numpy array avec les valeurs de toutes les metrics: Accuracy micro, mean Accuracy macro
    """
    metrics_value = []
    argmax_pred = torch.argmax(y_pred, dim=1)
    metrics_value += accuracy_without_pad(argmax_pred, y_true, num_classes)
    return np.array(metrics_value)


def accuracy_without_pad(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> list:
    """
    y_pred : [B, C, k]
    y_true : [B, k]
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


if __name__ == "__main__": 
    y_pred = torch.rand((10, 19, 10))
    y_true = torch.randint(0, 19, (10, 10))
    ic(compute_metrics(y_pred, y_true, 19))