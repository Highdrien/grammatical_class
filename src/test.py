import torch
from tqdm import tqdm
from easydict import EasyDict
from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
from src.metrics import compute_metrics
import matplotlib.pyplot as plt
from config.config import train_logger, train_step_logger
import time

def test(config: EasyDict,checkpoint_name='get_pos_lstm_1') -> None:
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get test data
    test_generator, _ = create_dataloader(config=config, mode='test')
    #test du dataloader
    #affiche le premier élément du dataloader, sa shape et son type
    print("Shape du batch:",next(iter(test_generator))[0].shape)
    print("First element du batch:",next(iter(test_generator))[0][0])
    print("Shape of the first element du batch:",next(iter(test_generator))[0][0].shape)



    n_test = len(test_generator)

    # Get model
    model = get_model(config)
    model = model.to(device)

    # Loss
    assert config.learning.loss == 'crossentropy', NotImplementedError(
        f"The loss '{config.learning.loss}' was not implemented. Only 'crossentropy' is implemented")
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    #get path to the trained model
    path='logs/'+checkpoint_name+'/checkpoint.pt' 

    # Load the trained weights
    model.load_state_dict(torch.load(path))
    print("Model loaded successfully!")

    # Testing
    model.eval()
    test_loss = 0
    test_metrics = 0
    test_range = tqdm(test_generator)

    with torch.no_grad():
        for x,y_true in test_range:

            print("shape of x:",x.shape)
            print("shape of y_true:",y_true.shape)
            x = x.to(device)
            #y_true = y_true.to(device)

            y_pred = model.forward(x)
            y_pred = y_pred.permute(0, 2, 1)

            #y_pred back to cpu
            y_pred=y_pred.cpu()

            test_metrics += compute_metrics(y_pred, y_true, config.task.get_pos_info.num_classes)

            test_range.refresh()

    test_metrics = test_metrics / n_test
    print(f"Test Metrics: {test_metrics[0]:.4f}")

    # Optionally, you can save the test results or generate plots if needed.

if __name__ == "__main__":
    # Assuming you have a config object for testing, update this accordingly
    test_config = EasyDict({
        # Add your test configuration parameters here
    })

    test(test_config)
