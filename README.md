But de ce repo:
avoir les Traits morphologiques pour chaque mots à partir de phrase.

# Data:

- find the data here: http://hdl.handle.net/11234/1-5287
- Download all files in item
- unzip it
- use `tar -xvzf /path/to/yourfile.tgz` to unstack the tgz file
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
  language: English           # Language of data
  sequence_length: 10         # length of sequence
  pad: <PAD>                  # pad caracter to make padding
  unk: <UNK>                  # unknow caracter
  sequence_function: dummy    # function to split sentences to sequences
  indexes: [1, 3]             # index of information that will be recupered: 
                                # indexes representation: 
                                # 0: id       1: word    2: lemma   3: pos   4: unk
                                # 5: morphy   6: syntax  7: unk     8: unk   9: unk 
  vocab:
    path: dictionary          # path to save the vocabulary
    unk_rate: 0.01            # rate to replace a knowing word by <UNK>
    save: true                # save the vocabulary or load it from data.voc.path
```


