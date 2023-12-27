import torch
import numpy as np
from typing import List, Any
from icecream import ic
from easydict import EasyDict
from torchmetrics import Accuracy


# def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> np.ndarray: 
#     """
#     On renvoie un numpy array avec les valeurs de toutes les metrics: Accuracy micro, mean Accuracy macro
#     """
#     metrics_value = []
#     argmax_pred = torch.argmax(y_pred, dim=1)
#     metrics_value += accuracy_without_pad(argmax_pred, y_true, num_classes)
#     return np.array(metrics_value)


# def accuracy_without_pad(y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int) -> list:
#     """
#     y_pred : [B, C, K] (C = num_classes, K = max_len)
#     y_true : [B, K]
#     On récupère la micro et la moyenne du macro
#     """
#     pad_index = 10 # A ne pas considérer du y_true
#     mask = (y_true != pad_index ) 
#     masked_y_pred = y_pred[mask]
#     masked_y_true = y_true[mask]
#     accuracy_macro = MulticlassAccuracy(num_classes=num_classes, average="macro")
#     accuracy_macro.update(masked_y_pred, masked_y_true)
#     accuracy_micro = MulticlassAccuracy(num_classes=num_classes, average="micro")
#     accuracy_micro.update(masked_y_pred, masked_y_true)
#     macro = torch.mean(accuracy_macro.compute()).item()
#     micro = accuracy_micro.compute().item()
#     return [macro, micro]




# def compute_metrics(y_pred: torch.Tensor,
#                     y_true: torch.Tensor,
#                     task: str
#                     ) -> float:
#     """
#     Calcule l'accuracy entre y_true et y_pred sans considérer le padding.
#     y_pred : [B, L, C] (B = batch_size,L = sequence_lenght, C = num_classes)
#     y_true : [B, L, C] (B = batch_size,L = sequence_lenght, C = num_classes)
#     """
    
#     # Convert probabilities to class predictions
#     if task == 'get_morphy':
#         y_pred_classes = torch.argmax(y_pred, dim=2)
    
#     # Convert one-hot encoded labels to class indices
#     y_true_classes = torch.argmax(y_true, dim=2)
    
#     # Compute accuracy
#     correct_predictions = torch.eq(y_pred_classes, y_true_classes)
#     accuracy_value = correct_predictions.sum().item() / correct_predictions.numel()

#     return accuracy_value


class Metrics:
    def __init__(self, config: EasyDict) -> None:
        if 'metrics' in config.keys():
            self.metrics_name = list(filter(lambda x: config.metrics[x], config.metrics))
        else:
            self.metrics_name = []
        
        self.metric = {}

    
    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        raise NotImplementedError
    
    def get_metrics_name(self) -> List[str]:
        raise NotImplementedError


class POS_Metrics(Metrics):
    def __init__(self, config: EasyDict, device: torch.device=None) -> None:
        super().__init__(config)

        num_classes = config.task[f'{config.task.task_name}_info'].num_classes

        self.metric : dict[str, Any] = {
            'acc micro': Accuracy(num_classes=num_classes, average='micro', task='multiclass'),
            'acc macro': Accuracy(num_classes=num_classes, average='macro', task='multiclass')
        }
        if device is not None:
            for key, value in self.metric.items():
                self.metric[key] = value.to(device)
    
    def compute(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
        """ compute metrics
        y_true: torch tensor (index) with a shape of (B, K)
        y_pred: torch.tensor (one hot) with a shape of (B, K, C)
        """
        y_pred_argmax = torch.argmax(y_pred, dim=2)
        metrics_value = []

        if 'acc' in self.metrics_name:
            micro = self.metric['acc micro'](y_pred_argmax, y_true)
            metrics_value.append(micro.item())
        
            macro = self.metric['acc macro'](y_pred_argmax, y_true)
            metrics_value.append(macro.item())
        
        return np.array(metrics_value)

    def get_metrics_name(self) -> List[str]:
        metrics_name = []
        if 'acc' in self.metrics_name:
            metrics_name += ['acc micro', 'acc macro']
        return metrics_name
        



if __name__ == "__main__":
    import yaml
    B = 256     # batch size
    K = 10      # sequence length
    V = 3000    # vocab size

    # mode: get_pos
    C = 19  # num classes
    x = torch.randint(0, V, (B, K))
    y_pred = torch.rand((B, K, C))
    y_true = torch.randint(0, C, (B, K))

    ic(x.shape, x.dtype)
    ic(y_pred.shape, y_pred.dtype)
    ic(y_true.shape, y_true.dtype)
    
    
    config = EasyDict(yaml.safe_load(open('config/config.yaml', 'r')))
    metric = POS_Metrics(config=config)
    metrics_value = metric.compute(y_pred=y_pred, y_true=y_true)
    ic(metrics_value)
