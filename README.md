But de ce repo:
avoir les Traits morphologiques pour chaque mots à partir de phrase.

# Data:

- find the data here: http://hdl.handle.net/11234/1-5287
- Download all files in item
- unzip it
- keep only the file `ud-treebanks-v2.13.tgz`
- use `tar -xvzf ud-treebanks-v2.13.tgz` to unstack the tgz file
- and move data in order have a path like:

   .                     
    ├── ...                                     
    ├── data                                   
    │   ├── UD_Abaza-ATD                              
	│   │   ├── abd_atb-ud-test.conllu                     
	│   │   ├── abd_atb-ud-test.txt        
	│   │   └── ...                        
    │   ├── UD_Afrikaans-AfriBooms                               
	│   │   ├── af_afribooms-ud-dev.conllu                      
	│   │   ├── af_afribooms-ud-dev.txt                         
	│   │   ├── af_afribooms-ud-test.conllu                      
	│   │   ├── af_afribooms-ud-test.txt                                 
	│   │   ├── af_afribooms-ud-train.conllu                     
	│   │   ├── af_afribooms-ud-train.txt             
	│   │   └── ...                                             
    │   ├── ...                      
    │   └── ...                                  


# Dataloader:

have data configuration like:
```yaml
data:
  path: data                  # data path
  language: French            # Language of data
  sequence_length: 10         # length of sequence
  pad: <PAD>                  # pad caracter to make padding
  unk: <UNK>                  # unknow caracter
  sequence_function: dummy    # function to split sentences to sequences
  indexes: [1, 3, 5]          # index of information that will be recupered: 
                                # indexes representation: 
                                # 0: id       1: word    2: lemma   3: pos   4: unk
                                # 5: morphy   6: syntax  7: unk     8: unk   9: unk 
  vocab:
    path: dictionary          # path to save the vocabulary
    unk_rate: 0.01            # rate to replace a knowing word by <UNK>
    save: true                # save the vocabulary or load it from data.voc.path
    num_words: 67814
```


## Create sequences:
'dummy': This is the most basic way. We cut the sentences without overlapping and add a completion character at the end to get the right size. For example, the sentence:\
   ['`I`', '`think`', '`,`', '`threfore`', '`I`', '`am`', '`.`']\
    will become\
   [['`I`', '`think`', '`,`', '`threfore`'], ['`I`', '`am`', '`.`', '`PAD`']]\
   if sequence_size is 4 and the completion character is '`PAD`'. 

Sure, here's an updated Markdown paragraph for your README file:

---

## Using the Main Script

The [`main.py`](command:_github.copilot.openRelativePath?%5B%22main.py%22%5D "main.py") script is the main entry point for this project. It accepts several command-line arguments to control its behavior:

- `--mode` or `-m`: This option allows you to choose a mode between 'train', 'baseline', 'test', and 'infer'.
- [`--config_path`](command:_github.copilot.openSymbolInFile?%5B%22main.py%22%2C%22--config_path%22%5D "main.py") or [`-c`](command:_github.copilot.openSymbolInFile?%5B%22src%2Ftrain.py%22%2C%22-c%22%5D "src/train.py"): This option allows you to specify the path to the configuration file for training. The default is [`config/config.yaml`](command:_github.copilot.openRelativePath?%5B%22config%2Fconfig.yaml%22%5D "config/config.yaml").
- [`--path`](command:_github.copilot.openSymbolInFile?%5B%22main.py%22%2C%22--path%22%5D "main.py") or `-p`: This option allows you to specify the experiment path for testing, prediction, or generation.
- [`--task`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmetrics.py%22%2C%22--task%22%5D "src/metrics.py") or `-t`: This option allows you to specify the task for the model. This will overwrite the task specified in the configuration file for training.

Here's what each mode does:

- [`train`](command:_github.copilot.openSymbolInFile?%5B%22src%2Ftrain.py%22%2C%22train%22%5D "src/train.py"): Trains a model using the configuration specified in the [`--config_path`](command:_github.copilot.openSymbolInFile?%5B%22main.py%22%2C%22--config_path%22%5D "main.py") and the task specified in [`--task`](command:_github.copilot.openSymbolInFile?%5B%22src%2Fmetrics.py%22%2C%22--task%22%5D "src/metrics.py").
- `baseline`: Runs the baseline benchmark test.
- `test`: Tests the model specified in the [`--path`](command:_github.copilot.openSymbolInFile?%5B%22launch_inference.py%22%2C%22--path%22%5D "launch_inference.py"). You must specify a path.
- [`infer`](command:_github.copilot.openRelativePath?%5B%22infer%22%5D "infer"): Runs inference using the model specified in the [`--path`](command:_github.copilot.openSymbolInFile?%5B%22launch_inference.py%22%2C%22--path%22%5D "launch_inference.py") It will run inference on some exemple sentences located in "infer/infer.txt" and put the results in "infer/configname_infer.txt". If the path is 'baseline', it will run the baseline inference. You must specify a path.

Here's an example of how to use the script to train a model:

```sh
python main.py --mode train --config_path config/config.yaml --task get_pos
```
Here is an exemple of how to run inference using the baseline model:
```sh
python main.py --mode infer --path baseline
```
This command will train a model using the configuration specified in [`config/config.yaml`](command:_github.copilot.openRelativePath?%5B%22config%2Fconfig.yaml%22%5D "config/config.yaml") and the task 'get_pos'.

---

Please replace the example paths and tasks with the actual paths and tasks you want to use in your project.




