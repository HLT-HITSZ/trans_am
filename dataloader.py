from torch.utils.data import Dataset
import pandas as pd
import json
import torch
from transformers import BertTokenizer
from config import get_config
config = get_config()
import sys
sys.path.append('./utils')
from transition_action import text_2_action_sequence
import numpy as np


class ArgMiningDataset(Dataset):
    
    def __init__(self, data_df, vocab_dict, dataset_name, mode):
        AC_type2id = config.AC_type2id
        para_type2id = config.para_type2id
        self.vocab_dict = vocab_dict
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

        para_text_list = list(data_df['para_text'])
        orig_AC_spans_list = [eval(AC_spans) for AC_spans in data_df['adu_spans']]
        AC_bow_token_ids_lists = []
        for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
            para_tokens = para_text.split(' ')
            para_token_ids = [vocab_dict[token] if token in vocab_dict \
                                                else vocab_dict['<unk>'] \
                                                for token in para_tokens]
            AC_token_ids_list = []
            for AC_span in AC_spans:
                AC_token_ids = para_token_ids[AC_span[0]:AC_span[1]+1]
                AC_token_ids_list.append(AC_token_ids)
            AC_bow_token_ids_lists.append(AC_token_ids_list)

        bow_lists = []
        eye = torch.tensor(np.identity(len(self.vocab_dict), dtype=np.float32))
        for AC_bow_token_ids_list in AC_bow_token_ids_lists:
            bow_list = []
            for AC_bow_token_ids in AC_bow_token_ids_list:
                bow_list.append(torch.sum(eye[AC_bow_token_ids], dim=0))
            bow_lists.append(torch.stack(bow_list))
        self.AC_bow_array_list = bow_lists

        self.special_tokens = {'<pad>', '<essay>', '<para-conclusion>', 
                        '<para-body>', '<para-intro>', '<ac>',
                        '</essay>', '</para-conclusion>', '</para-body>', 
                        '</para-intro>', '</ac>'}
        
        AC_spans_list = []
        para_token_ids_list = []
        for para_text, AC_spans in zip(para_text_list, orig_AC_spans_list):
            orig_pos2bert_pos = {}
            para_tokens = para_text.split(' ')
            para_tokens_for_bert = []
            para_tokens_for_bert = ['[CLS]']
            for orig_pos, token in enumerate(para_tokens):
                if token not in self.special_tokens:
                    if dataset_name == 'PE':
                        bert_tokens = self.tokenizer.tokenize(token)
                    elif dataset_name == 'CDCP':
                        bert_tokens = [token]
                else:
                    bert_tokens = [token]
                cur_len = len(para_tokens_for_bert)
                orig_pos2bert_pos[orig_pos] = (cur_len, cur_len+len(bert_tokens)-1)
                para_tokens_for_bert += bert_tokens
                if orig_pos == 509:
                    break
            para_tokens_for_bert.append('[SEP]')
            para_token_ids_list.append(self.tokenizer.convert_tokens_to_ids(para_tokens_for_bert))
            
            AC_spans_for_bert = []
            for AC_span in AC_spans:
                start = orig_pos2bert_pos[AC_span[0]][0]
                end = orig_pos2bert_pos[AC_span[1]][1]
                AC_spans_for_bert.append((start, end))
            AC_spans_list.append(AC_spans_for_bert)
        self.AC_spans_list = AC_spans_list
        self.para_token_ids_list = para_token_ids_list

        self.parser_states_list = text_2_action_sequence(data_df, dataset_name, mode)
        data_df['actions'] = self.parser_states_list

        para_types_list = list(data_df["para_types"])
        self.AC_types_list = [list(map(lambda x: AC_type2id[x], eval(AC_types))) \
                                for AC_types in data_df["AC_types"]]

        AC_positions_list = []
        AC_para_types_list = []

        for AC_spans, para_type in zip(self.AC_spans_list, para_types_list):
            AC_positions = list(range(len(AC_spans)))
            AC_positions_list.append(AC_positions)
            AC_para_types_list.append([para_type2id[para_type]] * len(AC_spans))
        self.AC_positions_list = AC_positions_list
        self.AC_para_types_list = AC_para_types_list

        self.AR_pairs_list = [eval(_) for _ in data_df['AR_pairs']]
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        one_sample = (
            self.para_token_ids_list[index],
            self.AC_bow_array_list[index],
            self.AC_spans_list[index],
            self.AC_positions_list[index],
            self.AC_para_types_list[index],
            self.parser_states_list[index],
            self.AC_types_list[index],
            self.AR_pairs_list[index]
        )
        return one_sample

def generate_batch_fn(batch):
    batch = list(zip(*batch))
    batch = {
        'para_tokens_ids': batch[0],
        'bow_vecs': batch[1],
        'AC_spans': batch[2],
        'AC_positions': batch[3],
        'AC_para_types': batch[4],
        'true_parser_states': batch[5], 
        'true_AC_types': batch[6],
        'true_AR_pairs': batch[7]
    }
    return batch