import torch
from tqdm import tqdm
from easydict import EasyDict
from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
from src.metrics import compute_metrics
import matplotlib.pyplot as plt
from config.config import train_logger, train_step_logger

def test(config: EasyDict,checkpoint_name='get_pos_lstm_1') -> None:
    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Get test data
    _, test_generator = create_dataloader(config=config, mode='test')
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
        for x, y_true in test_range:
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model.forward(x)
            y_pred = y_pred.permute(0, 2, 1)

            loss = criterion(y_pred, y_true)

            test_loss += loss.item()
            test_metrics += compute_metrics(y_pred, y_true, config.task.get_pos_info.num_classes)

            test_range.set_description(f"TEST  -> loss: {loss.item():.4f}")
            test_range.refresh()

    test_loss = test_loss / n_test
    test_metrics = test_metrics / n_test

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics[0]:.4f}")

    # Optionally, you can save the test results or generate plots if needed.

if __name__ == "__main__":
    # Assuming you have a config object for testing, update this accordingly
    test_config = EasyDict({
        # Add your test configuration parameters here
    })

    test(test_config)
