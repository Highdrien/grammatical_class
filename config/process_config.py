from easydict import EasyDict


def change_name(config: EasyDict) -> None:
    task_name = config.task.task_name
    model_name = config.model.model_name
    config.name = f"{task_name}_{model_name}" 


def analyse_config(config: EasyDict) -> None:
    indexes = config.data.indexes
    assert 1 in indexes, f"Error, 1 must be in indexes because it's the words"

    process_assert(possible=['dummy'],
                   value=config.data.sequence_function,
                   name_value='sequence function')
    
    process_assert(possible=['get_pos'],
                   value=config.task.task_name,
                   name_value='task name')
    
    process_assert(possible=['lstm', 'bert'],
                   value=config.model.model_name,
                   name_value='model name')
    
    process_assert(possible=['relu', 'sigmoid'],
                   value=config.model.lstm.activation,
                   name_value='activation function in lstm classifier')
    
    process_assert(possible=['crossentropy'],
                   value=config.learning.loss,
                   name_value='loss')
    
    process_assert(possible=['adam'],
                   value=config.learning.optimizer,
                   name_value='optimizer')
    


def process_assert(possible: list, value: any, name_value: str) -> None:
    error_message = f"Error, {name_value} must be in {possible}, but is '{value}'"
    assert value in possible, error_message


def process_config(config: EasyDict) -> None:
    analyse_config(config)
    change_name(config)