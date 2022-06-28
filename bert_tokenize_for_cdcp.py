import os
import torch
import sys
from transformers import BertTokenizer
import pandas as pd
from collections import Counter

bert_path = "data/bert-base-uncased"
bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

if __name__ == "__main__":
    bert_path = "data/bert-base-uncased"
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

    orig_data_path = "data/cdcp_data_df.csv"
    orig_data_df = pd.read_csv(orig_data_path)
    orig_text_list = list(orig_data_df["para_text"])
    tokenized_text_list = []
    new_adu_spans_list = []
    for orig_text in orig_text_list:
        tokenized_text_tokens = []
        adu_spans = []
        start = 0
        end = 0
        token_id = 0
        for word in orig_text.split(' '):
            if word == '<para-body>' or word == '</para-body>':
                pass
                token_id += 1
                tokenized_text_tokens.append(word)
            elif word == '<ac>':
                start = token_id
                token_id += 1
                tokenized_text_tokens.append(word)
            elif word == '</ac>':
                end = token_id
                adu_spans.append((start, end))
                token_id += 1
                tokenized_text_tokens.append(word)
            else:
                tokens = bert_tokenizer.tokenize(word)
                token_id += len(tokens)
                tokenized_text_tokens.extend(tokens)
        tokenized_text_list.append(' '.join(tokenized_text_tokens))
        new_adu_spans_list.append(adu_spans)
    orig_data_df["para_text"] = tokenized_text_list
    orig_data_df["adu_spans"] = new_adu_spans_list
    
    truncated_adu_spans_list = list(orig_data_df['adu_spans'])
    truncated_ac_types_list = list(map(eval, orig_data_df['AC_types']))
    truncated_ac_rel_pairs_list = list(map(eval, orig_data_df['AR_pairs']))

    droped_ac = 0
    droped_rel = 0

    for idx, (para_text, adu_spans, ac_types, ac_rel_pairs) \
        in enumerate(zip( orig_data_df['para_text'], truncated_adu_spans_list, 
                truncated_ac_types_list, truncated_ac_rel_pairs_list)):
        if len(para_text.split(' ')) > 510:
            exceeded_idx = len(adu_spans)
            for i, span in enumerate(adu_spans):
                if span[0] > 510 or span[1] > 510:
                    exceeded_idx = i
                    break
            new_adu_spans = adu_spans[:exceeded_idx]
            new_ac_types = ac_types[:exceeded_idx]
            new_ac_rel_pairs = []
            for pairs in ac_rel_pairs:
                if pairs[0] < exceeded_idx and pairs[1] < exceeded_idx:
                    new_ac_rel_pairs.append(pairs)
                else:
                    droped_rel += 1
            truncated_adu_spans_list[idx] = new_adu_spans
            truncated_ac_types_list[idx] = new_ac_types
            truncated_ac_rel_pairs_list[idx] = new_ac_rel_pairs
    orig_data_df['orig_adu_spans'] = orig_data_df['adu_spans']
    orig_data_df['orig_AC_types'] = orig_data_df['AC_types']
    orig_data_df['orig_AR_pairs'] = orig_data_df['AR_pairs']
    orig_data_df['adu_spans'] = truncated_adu_spans_list
    orig_data_df['AC_types'] = truncated_ac_types_list
    orig_data_df['AR_pairs'] = truncated_ac_rel_pairs_list

    orig_data_df.to_csv("data/cdcp_data_bert_df.csv", index=False)

    all_word_list = [word for para_text in orig_data_df["para_text"] \
                            for word in para_text.split(' ')]
    word_cnt = Counter(all_word_list)

    word2id = {'<pad>': 0, '<unk>': 1}
    for word, cnt in word_cnt.items():
        if cnt > 1:
            word2id[word] = len(word2id)

    import json
    with open('data/bow_vocab_cdcp.json', "w") as fp:
        json.dump(word2id, fp)




