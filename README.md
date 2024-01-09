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

## Model Training
# Pos-tagging training
In the config.yaml file make sure you have this: 
```yaml
task options
task:
    task_name: get_pos     # task do to: must be get_pos or get_morphy
    get_pos_info:
        num_classes: 19         # number of POS classes
    get_morphy_info:
        num_classes: 28         # number of MORPHY classes
        num_features: 13        # number of max classes possibility
        use_pos: true              # use POS to get MORPHY
```

Then run the launch_train.py file to launch training of model with this configuration.

# Morphological traits training
In the config.yaml file make sure you have this: 
```yaml
task options
task:
    task_name: get_morphy     # task do to: must be get_pos or get_morphy
    get_pos_info:
        num_classes: 19         # number of POS classes
    get_morphy_info:
        num_classes: 28         # number of MORPHY classes
        num_features: 13        # number of max classes possibility
        use_pos: true              # use POS to get MORPHY
```

Then run the launch_train.py file to launch training of model with this configuration.

## Test 
# Run test on pos-tagging


To run inference using the [`launch_inference.py`](command:_github.copilot.openRelativePath?%5B%22launch_inference.py%22%5D "launch_inference.py") script, follow these steps:

1. Ensure that you have a dictionary file in the [`dictionary/`](command:_github.copilot.openRelativePath?%5B%22dictionary%2F%22%5D "dictionary/") directory. For example, you might have a file named `French.json`.

2. Set the [`experiment_path`](command:_github.copilot.openSymbolInFile?%5B%22launch_inference.py%22%2C%22experiment_path%22%5D "launch_inference.py") variable to the path of the experiment you want to run inference on. For example, you might set it to [`logs/get_pos_French`](command:_github.copilot.openRelativePath?%5B%22logs%2Fget_pos_French%22%5D "logs/get_pos_French").

3. Define the [`sentence`](command:_github.copilot.openSymbolInFile?%5B%22launch_inference.py%22%2C%22sentence%22%5D "launch_inference.py") variable as a list of words for which you want to run inference. For example:

```python
sentence = ['je', 'veux', 'un','chien', '.','Mais ça','narrivera','pas','.','Je','nai','pas','argent','.']
```

4. Run the [`launch_inference.py`](command:_github.copilot.openRelativePath?%5B%22launch_inference.py%22%5D "launch_inference.py") script. The script will load the dictionary, run inference on the sentence using the specified experiment, and print the sentence and its part-of-speech tags.

Here's an example of how to run the script:

```sh
python launch_inference.py
```

The output will look something like this:

```
sentence: ['je', 'veux', 'un', 'chien', '.', 'Mais ça', 'narrivera', 'pas', '.', 'Je', 'nai', 'pas', 'argent', '.']
POS: ['PRON', 'VERB', 'DET', 'NOUN', 'PUNCT', 'CONJ', 'VERB', 'ADV', 'PUNCT', 'PRON', 'VERB', 'ADV', 'NOUN', 'PUNCT']
```

This indicates that the word 'je' is a pronoun, 'veux' is a verb, 'un' is a determiner, and so on.

---

# Run test on morphological traits

Running for morphological traits is the same as for pos-tagging, except that you need to set the ['experiment_path'](command:_github.copilot.openSymbolInFile?%5B%22launch_inference.py%22%2C%22experiment_path%22%5D "launch_inference.py") variable to the path of the experiment you want to run inference on. For example, you might set it to ['logs/separate'](command:_github.copilot.openRelativePath?%5B%22logs%2Fget_morphy_French%22%5D "logs/get_morphy_French") or ['logs/fusion'](command:_github.copilot.openRelativePath?%5B%22logs%2Fget_morphy_French%22%5D "logs/separate") or ['logs/supertag'](command:_github.copilot.openRelativePath?%5B%22logs%2Fget_morphy_French%22%5D "logs/fusion").

The output will look something like this:

```sh
WORD: je
OUTPUT: ['Not', 'Not', 'No', 'Neut', 'Sing', '1', 'Emp', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Sup', 'Not', 'Not', 'Not', 'Digit', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not']
WORD: veux
OUTPUT: ['Not', 'Not', 'Not', 'Not', 'Sing', '3', 'Tot', 'Not', 'Not', 'Fut', 'Fin', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Vrnc', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not', 'Not']
WORD: un
OUTPUT: ['Not', 'Not', 'Not', 'Neut', 'Sing', 'Not', 'Emp', 'Not', 'Not', 'Not', 'Not', 'Mult', 'Not', 'Not', 'Not', 'Not', 'Not', 'Sup', 'Arch', 'Not', 'Not', 'Digit', 'Not', 'Not', 'Not', 'Not', 'Ind', 'Not']
...
```





