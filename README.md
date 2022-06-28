Code for our paper: 


Jianzhu Bao, Chuang Fan, Jipeng Wu, Yixue Dang, Jiachen Du, and Ruifeng Xu. 2021. A Neural Transition-based Model for Argumentation Mining. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6354–6364, Online. Association for Computational Linguistics.


# Prerequisites
python 3.6  
pytorch 1.7.1  
cuda 10.2  
transformers 4.2.1  
json 2.0.9  
numpy 1.19.4  
pandas 1.1.5  
scikit-learn 0.24.0  
tqdm 4.54.1  
argparse 1.1
# Descriptions
**data** - contains dataset.  
* ```bert-base-uncased```: put the download Pytorch bert model here (config.json, pytorch_model.bin, vocab.txt) (https://huggingface.co/bert-base-uncased/tree/main). 
* ```acl2017-neural_end2end_am```: contains PE dataset, put "data/" directory of (https://github.com/UKPLab/acl2017-neural_end2end_am) into "data/acl2017-neural_end2end_am/".
* ```cdcp```: contains CDCP dataset, put "cdcp/" directory in (https://facultystaff.richmond.edu/~jpark/data/cdcp_acl17.zip) into "data/".
* ```test_paragraph_index.json```: contains test set of PE dataset, adopted from (https://github.com/kuribayashi4/span_based_argumentation_parser/blob/master/work/test_paragraph_index.json).
* ```cdcp_test_index.json```: contains test set of CDCP dataset, derived from "cdcp_acl17.zip"

**saved_models** - contains saved models, training logs and results.  

**utils** - utils code.  
* ```config.py```: parameter setting. 
* ```evaluation.py```: evaluation procedure.
* ```dataloader.py``` - load train, dev, test data.  
* ```trans_module.py```: proposed transition-based model.
* ```transitions.py```: transform text into transitions.

**preprocess_cdcp** - utils for process cdcp dataset, adopted from (https://github.com/vene/marseille).


```load_text_essays.py```: load PE dataset. 
```prepare_data.py``` - preprocess the input data.  
```run.py``` - train and evaluate the proposed transition-based model.  
```preprocess.sh``` - prepare data and models.  

# Usage

## For PE dataset


bash process_pe.sh
python run.py --dataset-name PE --data-path ./data/PE_data_df.csv --split-test-file-path ./data/test_paragraph_index.json --vocab-path ./data/bow_vocab.json --batch-size 32 --showtime 2 --early-num 5 --max-AC-num 50 --AC-type-label-num 3


## For CDCP dataset


bash process_cdcp.sh
python run.py  --dataset-name CDCP --data-path ./data/cdcp_data_bert_df.csv --split-test-file-path ./data/cdcp_test_index.json --vocab-path ./data/bow_vocab_cdcp.json --batch-size 16 --showtime 10 --early-num 10 --max-AC-num 12 --AC-type-label-num 5