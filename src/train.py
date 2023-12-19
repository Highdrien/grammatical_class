import os
import torch
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import time
from easydict import EasyDict
from icecream import ic

from src.dataloader.dataloader import create_dataloader
from src.model.get_model import get_model
from config.config import train_logger, train_step_logger

from src.metrics import compute_metrics, accuracy_without_pad
import matplotlib.pyplot as plt



def annotate_curve(config: EasyDict):
    """
    Annotate the curve of training by creating a str synthetizing the config file.
    return: str
    """
    str_annotate = ""
    str_annotate += f"model: {config.model.model_name}\n"
    str_annotate += f"loss: {config.learning.loss}\n"
    str_annotate += f"optimizer: {config.learning.optimizer}\n"
    str_annotate += f"learning_rate: {config.learning.learning_rate}\n"
    str_annotate += f"epochs: {config.learning.epochs}\n"
    str_annotate += f"batch_size: {config.learning.batch_size}\n"
    str_annotate += f"gamma: {config.learning.gamma}\n"
    
    return str_annotate
    

def train(config: EasyDict) -> None:

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    device = torch.device("cpu")
    ic(device)

    # Get data
    train_generator, _ = create_dataloader(config=config, mode='train')
    val_generator, _ = create_dataloader(config=config, mode='val')
    n_train, n_val = len(train_generator), len(val_generator)
    ic(n_train, n_val)

    # Get model
    model = get_model(config)
    model = model.to(device)
    ic(model)
    ic(model.get_number_parameters())
    
    # Loss
    assert config.learning.loss == 'crossentropy', NotImplementedError(
        f"The loss '{config.learning.loss}' was not implemented. Only 'crossentropy' is inplemented")
    criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    # Optimizer and Scheduler
    assert config.learning.optimizer == 'adam', NotImplementedError(
        f"The optimizer '{config.learning.optimizer}' was not implemented. Only 'adam' is inplemented")
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=config.learning.milesstone, gamma=config.learning.gamma)

    save_experiment = config.save_experiment
    ic(save_experiment)
    if save_experiment:
        logging_path = train_logger(config)
        best_val_loss = 10e6


    ###############################################################
    # Start Training                                              #
    ###############################################################
    start_time = time.time()

    list_train_loss = [] #liste des loss moyennes sur les epochs (train)
    list_acc_loss = []  #liste des métriques moyennes sur les epochs (train)
    list_val_loss = [] #liste des loss moyennes sur les epochs (validation)
    list_val_metric = [] #liste des métriques moyennes sur les epochs (validation)

    for epoch in range(1, config.learning.epochs + 1):
        ic(epoch)
        train_loss = 0
        train_metrics = 0
        train_range = tqdm(train_generator)

        # Training
        for x, y_true in train_range:
            
            x = x.to(device)
            y_true = y_true.to(device)

            y_pred = model.forward(x)
            y_pred = y_pred.permute(0, 2, 1)

            loss = criterion(y_pred, y_true)

            train_loss += loss.item() #calcul de la somme des loss sur le batch
            train_metrics += compute_metrics(y_pred, y_true, config.task.get_pos_info.num_classes) #calcul de la somme des métriques sur le batch

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_range.set_description(f"TRAIN -> epoch: {epoch} || loss: {loss.item():.4f}")
            train_range.refresh()

        train_loss = train_loss / n_train #calcul de la loss moyenne sur l'epoch
        train_metrics = train_metrics / n_train #calcul de la métrique moyenne sur l'epoch

        list_train_loss.append(train_loss) #ajout de la loss moyenne sur l'epoch à la liste des loss
        list_acc_loss.append(train_metrics[0]) #ajout de la métrique moyenne sur l'epoch à la liste des métriques



        ###############################################################
        # Start Validation                                            #
        ###############################################################

        val_loss = 0
        val_metrics = 0
        val_range = tqdm(val_generator)
        

        with torch.no_grad():

            val_loss = 0
            val_metrics = 0
            
            for x, y_true in val_range:
                x = x.to(device)
                y_true = y_true.to(device)

                y_pred = model.forward(x)
                y_pred = y_pred.permute(0, 2, 1)
                    
                loss = criterion(y_pred, y_true)


                # y_pred = torch.nn.functional.softmax(y_pred, dim=1)
                
                val_loss += loss.item() #calcul de la loss moyenne sur le batch
                val_metrics += compute_metrics(y_pred, y_true, config.task.get_pos_info.num_classes) #calcul de la métrique moyenne sur le batch

                val_range.set_description(f"VAL   -> epoch: {epoch} || loss: {loss.item():.4f}")
                val_range.refresh()
        
        scheduler.step()

        val_loss = val_loss / n_val #calcul de la loss moyenne sur l'epoch
        val_metrics = val_metrics / n_val
        
        list_val_loss.append(val_loss) #ajout de la loss moyenne sur l'epoch à la liste des loss
        list_val_metric.append(val_metrics[0])

        ###################################################################
        # Save Scores in logs                                             #
        ###################################################################
        train_loss = train_loss / n_train
        val_loss = val_loss / n_val
        # train_metrics = train_metrics / n_train
        # val_metrics = val_metrics / n_val
        
        if save_experiment:
            train_step_logger(path=logging_path, 
                              epoch=epoch, 
                              train_loss=train_loss, 
                              val_loss=val_loss)
            
            if config.learning.save_checkpoint and val_loss < best_val_loss:
                print('save model weights')
                torch.save(model.state_dict(), os.path.join(logging_path, 'checkpoint.pt'))
                best_val_loss = val_loss
                ic(best_val_loss)

    stop_time = time.time()
    print(f"training time: {stop_time - start_time}secondes for {config.learning.epochs} epochs")
    #plot loss

    annotation=annotate_curve(config) #création de la str d'annotation
    print("ANNOTATION", annotation)

    plt.figure()
    plt.plot(list_train_loss, label='train loss')
    plt.plot(list_val_loss, label='val loss')
    plt.annotate(annotation, xy=(0.5, 0.5), xycoords='axes fraction')
    plt.legend()
    plt.title('loss')
    plt.show()

    #plot accuracy
    plt.figure()
    plt.plot(list_acc_loss, label='train accuracy')
    plt.plot(list_val_metric, label='val accuracy')
    plt.annotate(annotation, xy=(0.5, 0.5), xycoords='axes fraction')
    plt.legend()
    plt.title('accuracy')
    plt.show()
