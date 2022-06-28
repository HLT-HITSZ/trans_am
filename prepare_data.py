import json
import numpy as np
import os, sys
from collections import defaultdict
import itertools
import datetime
import pandas as pd
import json
import random
import argparse
from collections import Counter

def get_data_dicts(data_path):

    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(SCRIPT_PATH,
                            data_path)

    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH)

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_essay_info_dict(FILENAME):

    with open(FILENAME) as f:
        n_span_para = [] # AC_id_in_paragraph
        n_span_essay = [] # AC_id_in_essay
        n_para = [] # Paragraph_id_in_essay
        for line in f:
            if line.split("\t")[6] != "-" \
                and line.split("\t")[5] != "AC_id_in_essay" \
                and line.split("\t")[6] != "AC_id_in_paragraph":
                n_span_essay.append(int(line.split("\t")[5]))
                n_span_para.append(int(line.split("\t")[6]))
                n_para.append(int(line.split("\t")[2]))

    max_n_spans = max(n_span_para) + 1
    max_n_paras = max(n_para) + 1

    essay_info_dict = {}
    split_column = 1

    essay2parainfo = defaultdict(dict)
    essay2paraids = defaultdict(list)
    para2essayid = dict()

    with open(FILENAME) as f:
        # for every paragraph; groupby can group words into group
        for essay_id, lines in itertools.groupby(f, key=lambda x: x.split("\t")[split_column]):
            if essay_id == "Essay_id" or essay_id == "Paragraph_id" or essay_id == "-":
                continue

            essay_lines = list(lines)
            para_type = essay_lines[0].split("\t")[3]
            essay_id = int(essay_lines[0].split("\t")[0])
            para_id = int(essay_lines[0].split("\t")[1])

            para2essayid[para_id] = essay_id
            essay2paraids[essay_id].append(para_id)
            essay2parainfo[essay_id][para_type] = para_id

            essay_info_dict[int(para_id)] = get_essay_detail(essay_lines,
                                                             max_n_spans)

    max_n_tokens = max([len(essay_info_dict[essay_id]["text"])
                        for essay_id in range(len(essay_info_dict))])

    essay_max_n_dict = {}
    essay_max_n_dict["max_n_spans_para"] = max_n_spans
    essay_max_n_dict["max_n_paras"] = max_n_paras
    essay_max_n_dict["max_n_tokens"] = max_n_tokens
    # for each para_id(key), the structure info of the essay it in(value)
    para_info_dict = defaultdict(dict)
    for para_id, essay_id in para2essayid.items():
        para_info_dict[para_id]["prompt"] = essay2parainfo[essay_id]["prompt"]
        para_info_dict[para_id]["intro"] = essay2parainfo[essay_id]["intro"]
        para_info_dict[para_id]["conclusion"] = essay2parainfo[essay_id]["conclusion"]
        para_info_dict[para_id]["context"] = essay2paraids[essay_id]

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_essay_detail(essay_lines, max_n_spans):
    essay_id = int(essay_lines[0].split("\t")[0])
    # list of (start_idx, end_idx) of each ac span
    ac_spans = []
    # text of each span
    ac_texts = []
    # type of each span (premise, claim, majorclaim)
    ac_types = []
    # in which type of paragraph each ac is (opening, body, ending)
    ac_paratypes = []
    # id of the paragraph where the ac appears
    ac_paras = []
    # id of each ac (in paragraoh)
    ac_positions_in_para = []
    # linked acs (source_ac, target_ac, relation_type)
    ac_relations = []
    # list of (startr_idx, end_idx) of each am span
    shell_spans = []
    relation2id = {"Support": 0, "Attack": 1}
    actype2id = {"Premise": 0, "Claim": 1, "Claim:For": 1, "Claim:Against": 1, "MajorClaim": 2}
    paratype2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq.fill(2)

    text = [line.strip().split("\t")[-1].lower()
        for line in essay_lines]

    previous_span_end = 0
    for ac_type, lines in itertools.groupby(essay_lines, key=lambda x: x.split("\t")[6]):
        ac_lines = list(lines)

        if ac_lines[0].split("\t")[7] != "-":
            ac_text = [ac_line.split("\t")[-1].strip() for ac_line in ac_lines]
            ac_texts.append(ac_text)

            para_i = int(ac_lines[0].split("\t")[2])
            para_type = ac_lines[0].split("\t")[3]
            ac_i = int(ac_lines[0].split("\t")[6])
            ac_type = ac_lines[0].split("\t")[7]
            start = int(ac_lines[0].split("\t")[11])
            end = int(ac_lines[-1].split("\t")[11])

            ac_positions_in_para.append(ac_i)
            ac_types.append(actype2id[ac_type])
            ac_paratypes.append(para_type)
            ac_paras.append(para_i)

            ac_span = (start, end)
            ac_spans.append(ac_span)

            shell_span = get_shell_lang_span(start, text, previous_span_end)
            shell_spans.append(shell_span)

            if ac_type == "Claim:For":
                relation_type_seq[ac_i] = 0
            elif ac_type == "Claim:Against":
                relation_type_seq[ac_i] = 1

            if "Claim" not in ac_lines[0].split("\t")[7]:
                ac_relations.append(
                   (ac_i,
                    ac_i + int(ac_lines[0].split("\t")[8]),
                    relation2id[ac_lines[0].split("\t")[9].strip()]))
                relation_type_seq[ac_i] = relation2id[ac_lines[0].split("\t")[9].strip()]
            previous_span_end = end

    assert len(ac_spans) == len(ac_positions_in_para)
    assert len(ac_spans) == len(ac_types)
    assert len(ac_spans) == len(ac_paratypes)
    assert len(ac_spans) == len(ac_paras)
    assert len(ac_spans) == len(shell_spans)
    assert len(relation_type_seq) == max_n_spans

    assert max(relation_type_seq).tolist() <= 2
    assert len(ac_spans) >= len(ac_relations)

    n_acs = len(ac_spans)
    relation_type_seq[n_acs:] = -1

    relation_targets, _, relation_depth = \
        relation_info2target_sequence(ac_relations,
                                      ac_types,
                                      max_n_spans,
                                      n_acs)

    assert len(relation_targets) == max_n_spans

    essay_detail_dict = {}
    essay_detail_dict["essay_id"] = essay_id
    essay_detail_dict["text"] = text
    essay_detail_dict["ac_spans"] = ac_spans
    essay_detail_dict["shell_spans"] = shell_spans
    essay_detail_dict["ac_types"] = np.pad(ac_types,
                                           [0, max_n_spans-len(ac_types)],
                                           'constant',
                                           constant_values=(-1, -1))
    essay_detail_dict["ac_paratypes"] = ac_paratypes
    essay_detail_dict["ac_paras"] = ac_paras
    essay_detail_dict["relation_targets"] = relation_targets
    essay_detail_dict["ac_relation_types"] = relation_type_seq

    return essay_detail_dict


def relation_info2target_sequence(ac_relations, ac_types, max_n_spans, n_spans):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        array: array of target ac index
    """

    relation_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    direction_seq = np.zeros(max_n_spans).astype('int32')
    depth_seq = np.zeros(max_n_spans).astype('int32')

    relation_seq.fill(max_n_spans)
    relation_type_seq.fill(2)
    direction_seq.fill(2)
    depth_seq.fill(100)
    relation_seq[n_spans:] = -1
    relation_type_seq[n_spans:] = -1
    direction_seq[n_spans:] = -1
    depth_seq[n_spans:] = -1

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_seq[source_i] = target_i
        relation_type_seq[source_i] = combination[2]

    for i in range(len(relation_seq)):
        depth = 0
        target_i = relation_seq[int(i)]
        if target_i == -1:
            continue
        while(1):
            if target_i == max_n_spans:
                break
            else:
                target_i = relation_seq[int(target_i)]
                depth += 1
        depth_seq[i] = depth

    return relation_seq, relation_type_seq, depth_seq

def get_shell_lang_span(start, text, previous_span_end):

    EOS_tokens_list = [".",
                       "!",
                       "?",
                       "</AC>",
                       "</para-intro>",
                       "</para-body>",
                       "</para-conclusion>",
                       "</essay>"]

    # EOS_ids_set = set([vocab[token.lower()
    #                    for token in EOS_tokens_list if token.lower() in vocab])
    EOS_ids_set = set([token.lower()
                       for token in EOS_tokens_list])
    shell_lang = []
    if start == 0:
        shell_span = (start, start)
        return shell_span

    for i in range(start-1, previous_span_end, -1):
        if text[int(i)] not in EOS_ids_set:
            shell_lang.append(int(i))
        else:
            break
    if shell_lang:
        shell_start = min(shell_lang)
        shell_end = max(shell_lang)
        shell_span = (shell_start, shell_end)
    else:
        shell_span = (start-1, start-1)
    return shell_span

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, \
                        default='data/PE_token_level_data.tsv')
    parser.add_argument("--data-save-path", type=str, \
                        default='data/PE_data_df.csv')
    parser.add_argument("--vocab-save-path", type=str, \
                        default='data/bow_vocab.json')
    args = parser.parse_args()

    essay_info_dict_list, essay_max_n_dict, para_info_dict = get_data_dicts(args.data_path)
    df_dict = { "essay_id": [],
                "para_id": [],
                "para_types": [],
                "para_text": [], 
                "adu_spans": [], 
                "ac_spans": [], 
                "ai_spans": [],
                "AC_types": [],
                "AR_pairs": []}
    punc_set = set([',', '?', '!', '.', '-', ';'])
    id2rel = {0: "Support", 1: "Attack", 2: "MC_rel"}
    id2ac_type = {0: "Premise", 1: "Claim", 2: "MajorClaim"}
    id2para_type = {0: "intro", 1: "body", 2: "conclusion", 3: "prompt"}
    text2para_type = {"<prompt>": "prompt", 
                      "<para-intro>": "intro",
                      "<para-body>": "body", 
                      "<para-conclusion>": "conclusion"}


    for global_para_id, para_info_dict in essay_info_dict_list.items():
        df_dict["para_text"].append(' '.join(para_info_dict["text"]))
        ac_spans = para_info_dict["ac_spans"]
        ai_spans = para_info_dict["shell_spans"]
        df_dict["ac_spans"].append(ac_spans)
        df_dict["ai_spans"].append(ai_spans)
        df_dict["para_types"].append(text2para_type[para_info_dict["text"][0]])
        adu_spans = []
        for ac_span, ai_span in zip(ac_spans, ai_spans):
            # if ai_span[0] == ai_span[1] \
            # and para_info_dict["text"][ai_span[0]] in punc_set:
            #     adu_span = ac_span
            # else:
            #     adu_span = (ai_span[0], ac_span[1])
            adu_span = (ai_span[0], ac_span[1])
            adu_spans.append(adu_span)
        df_dict["adu_spans"].append(adu_spans)
        AC_types = para_info_dict["ac_types"][:len(ac_spans)].tolist()
        ac_rel_targets = para_info_dict["relation_targets"][:len(ac_spans)].tolist()
        AC_types = [id2ac_type[type_id] for type_id in AC_types]
        ac_rel_targets = [target_id for target_id in ac_rel_targets]
        df_dict["AC_types"].append(AC_types)
        AR_pairs = []
        for child, parent in enumerate(ac_rel_targets):
            if parent != 12:
                AR_pairs.append((parent, child))
        df_dict["AR_pairs"].append(AR_pairs)
        df_dict["para_id"].append(global_para_id)
        df_dict["essay_id"].append(para_info_dict["essay_id"])

    pe_data_df = pd.DataFrame(df_dict)
    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_SAVE_PATH = os.path.join(SCRIPT_PATH,
                            args.data_save_path)
    pe_data_df.to_csv(DATA_SAVE_PATH, index=False)

    all_word_list = [word for para_text in pe_data_df["para_text"] \
                            for word in para_text.split(' ')]
    word_cnt = Counter(all_word_list)

    word2id = {'<pad>': 0, '<unk>': 1}
    for word, cnt in word_cnt.items():
        if cnt > 1:
            word2id[word] = len(word2id)

    import json
    with open(args.vocab_save_path, "w") as fp:
        json.dump(word2id, fp)