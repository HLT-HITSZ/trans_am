"""Run preprocessing on the data."""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

import os
import json
import sys
import numpy as np
sys.path.append('./preprocess_cdcp')
from collections import Counter
from marseille_datasets import get_dataset_loader
import pandas as pd
import re
# from marseille_argdoc import CdcpArgumentationDoc

def write_files(file, base_dir):
    for doc in file:
        with open(os.path.join(base_dir, "{:05d}.txt".format(doc.doc_id)),
                  "w") as f:
            f.write(doc.text)

        with open(os.path.join(base_dir, "{:05d}.ann.json".format(doc.doc_id)),
                  "w") as f:
            metadata = {'prop_offsets': doc.prop_offsets,
                        'prop_labels': doc.prop_labels,
                        'reasons': doc.reasons,
                        'evidences': doc.evidences,
                        'url': doc.url}
            json.dump(metadata, f, sort_keys=True)


def optimize_glove(glove_path, vocab):
    """Trim down GloVe embeddings to use only words in the data."""
    vocab_set = frozenset(vocab)
    seen_vocab = []
    X = []
    with open(glove_path) as f:
        for line in f:
            line = line.strip().split(' ')  # split() fails on ". . ."
            word, embed = line[0], line[1:]
            if word in vocab_set:
                X.append(np.array(embed, dtype=np.float32))
                seen_vocab.append(word)
    return seen_vocab, np.row_stack(X)


def store_optimized_embeddings(dataset, glove_path):

    out_path = os.path.join('data', '{}-glove.npz'.format(dataset))
    vocab = set()
    load, ids = get_dataset_loader(dataset, "train")
    for doc in load(ids):
        vocab.update(doc.tokens())
    res = optimize_glove(glove_path, vocab)
    glove_vocab, glove_embeds = res
    coverage = len(glove_vocab) / len(vocab)
    np.savez(out_path, vocab=glove_vocab, embeds=glove_embeds)

def _transitive(links, links_type):
    """perform transitive closure of links.

    For input [(1, 2), (2, 3)] the output is [(1, 2), (2, 3), (1, 3)]
    """
    # links_set = set(links)
    while True:
        new_links = set([(src_a, trg_b)
                     for src_a, trg_a in links
                     for src_b, trg_b in links
                     if trg_a == src_b
                     and (src_a, trg_b) not in links])
        if new_links:
            # links_set.update(new_links)
            links += list(new_links)
            # assert len(links_set) == len(links)
            links_type += ['reason'] * len(new_links)
        else:
            break

    return links, links_type


def process_punc(input_str):
    all_punc = {'*', '.', ';', '@', '!', '[', '_',\
                '#', ',', '%', '&', '$', '+',\
                '?', ':', '-', ']', '"', "'", '(',
                '/', ')'}
    for punc in all_punc:
        input_str = input_str.replace(punc, ' ' + punc + ' ')
    output_str = re.sub(" +", ' ', input_str).strip()
    return output_str


def get_cdcp_df_dict(dataset_type):
    load, ids = get_dataset_loader('cdcp', dataset_type)
    cdcp_df_dict = {
        'essay_id': [], 
        'para_text': [],
        'adu_spans': [],
        'ac_types': [], 
        'ac_rel_pairs': [],
        'ac_rel_types': [], 
        'dataset_type': [],
        'para_types': []
    }

    for doc in load(ids):
        tokens = []
        adu_spans = []
        start = 0
        end = 0
        tokens.append('<para-body>')
        start += 1
        end += 1
        for prop_s, prop_e in doc.prop_offsets:
            prop_text = process_punc(doc.text[prop_s:prop_e])
            tokens.append('<ac>')
            end += 1
            for tok in prop_text.split(' '):
                tokens.append(tok)
                end += 1
            tokens.append('</ac>') 
            end += 1
            adu_spans.append((start, end-1))
            start = end
        tokens.append('</para-body>')
        links, links_type = _transitive(doc.links, doc.links_type)
        links = [(link[1], link[0]) for link in links]
        
        cdcp_df_dict['essay_id'].append(doc.doc_id)
        cdcp_df_dict['para_text'].append(' '.join(tokens))
        cdcp_df_dict['adu_spans'].append(adu_spans)
        cdcp_df_dict['ac_types'].append(doc.prop_labels)
        cdcp_df_dict['ac_rel_pairs'].append(links)
        cdcp_df_dict['ac_rel_types'].append(links_type)
        cdcp_df_dict['dataset_type'].append(dataset_type)
        cdcp_df_dict['para_types'].append('body')
    return cdcp_df_dict

if __name__ == '__main__':
    train_cdcp_df_dict = get_cdcp_df_dict('train')
    test_cdcp_df_dict = get_cdcp_df_dict('test')

    cdcp_df_dict = {
        'essay_id': train_cdcp_df_dict['essay_id'] + test_cdcp_df_dict['essay_id'], 
        'para_id': train_cdcp_df_dict['essay_id'] + test_cdcp_df_dict['essay_id'], 
        'para_types': train_cdcp_df_dict['para_types'] + test_cdcp_df_dict['para_types'],
        'para_text': train_cdcp_df_dict['para_text'] + test_cdcp_df_dict['para_text'],
        'adu_spans': train_cdcp_df_dict['adu_spans'] + test_cdcp_df_dict['adu_spans'],
        'AC_types': train_cdcp_df_dict['ac_types'] + test_cdcp_df_dict['ac_types'], 
        'AR_pairs': train_cdcp_df_dict['ac_rel_pairs'] + test_cdcp_df_dict['ac_rel_pairs']
    }

    cdcp_df = pd.DataFrame(cdcp_df_dict)
    cdcp_df.to_csv('./data/cdcp_data_df.csv', index=False)    

    
