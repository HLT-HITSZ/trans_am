import sys
sys.path.append('../')
from sklearn.metrics import f1_score
import torch
from config import get_config
import itertools
config = get_config()

def get_eval_result(res_dict, dataset_name):
    if dataset_name == 'PE':
        ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format( \
                res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
        ACTC_msg = 'ACTC-Macro: {:.4f}\tMC: {:.4f}\tClaim: {:.4f}\tPremise: {:.4f}'.format( \
                res_dict["ACTC-Macro"], res_dict["MC"], res_dict["Claim"], res_dict["Premise"])
        macro_msg = 'Total: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}'.format( \
                res_dict["Total"], res_dict["ARI-Macro"], res_dict["ACTC-Macro"])
    elif dataset_name == 'CDCP':
        ARI_msg = 'ARI-Macro: {:.4f}\tRel: {:.4f}\tNo-Rel: {:.4f}'.format( \
            res_dict["ARI-Macro"], res_dict["Rel"], res_dict["No-Rel"])
        ACTC_msg = 'ACTC-Macro: {:.4f}\tValue: {:.4f}\tTestimony: {:.4f}\tPolicy: {:.4f}\tFact: {:.4f}\tReference: {:.4f}'.format( \
                res_dict["ACTC-Macro"], res_dict["Value"], res_dict["Testimony"], 
                res_dict["Policy"], res_dict["Fact"], res_dict["Reference"])
        macro_msg = 'Total: {:.4f}\tARI-Macro: {:.4f}\tACTC-Macro: {:.4f}'.format( \
                res_dict["Total"], res_dict["Rel"], res_dict["ACTC-Macro"]) 
    return ARI_msg, ACTC_msg, macro_msg


def get_no_rel_pairs(rel_pairs_list, ACs_num_list):
    no_rel_pairs_list = []
    for rel_pairs, ACs_num in zip(rel_pairs_list, ACs_num_list):
        rel_pairs_set = set(rel_pairs)
        all_AC_ids = list(range(ACs_num))
        all_pair_set = set([pair if pair[0] != pair[1] else None
                        for pair in itertools.product(all_AC_ids, all_AC_ids)])
        all_pair_set.remove(None)
        no_rel_pairs_set = all_pair_set - rel_pairs_set
        no_rel_pairs_list.append(no_rel_pairs_set)
    return no_rel_pairs_list


def pair_metric(preds, grounds):
    tn, fn, fp, tp = 0, 0, 0, 0
    for i in range(len(preds)):
        pred, ground = preds[i], grounds[i]
        t_pair = set(ground)            
        p_pair = set([x for x in pred if len(x) == 2])
        tp += len(p_pair & t_pair)
        fn += (len(t_pair) - len(p_pair & t_pair))
        fp += (len(p_pair) - len(p_pair & t_pair))
    if tp + fp == 0:
        pre = tp/(tp + fp + 1e-10)
    else:
        pre = tp/(tp + fp)
    if tp + fn == 0:
        rec = tp/(tp + fn + 1e-10)
    else:
        rec = tp/(tp + fn)
    if pre == 0. and rec == 0.:
        f1 = (2 * pre *rec)/(pre + rec + 1e-10)
    else:
        f1 = (2 * pre *rec)/(pre + rec)
    pr_metric = (pre, rec, f1)
    return pr_metric

def align_with_orig(all_pred_AC_types, AC_nums, orig_all_true_AC_types_para):
    all_pred_AC_types_para = []
    s = 0
    for num in AC_nums:
        all_pred_AC_types_para.append(all_pred_AC_types[s:s+num])
        s += num
    orig_all_pred_AC_types = []
    for idx, __ in enumerate(all_pred_AC_types_para):
        tmp = __ + [0]*(len(orig_all_true_AC_types_para[idx])-len(__))
        for _ in tmp:
            orig_all_pred_AC_types.append(_)
    return orig_all_pred_AC_types


def evaluate(trans_model, base_encoder, data_df, data_loader, dataset_name, mode="eval"):

    trans_model.eval()
    base_encoder.eval()
    all_pred_AR_pairs, all_true_AR_pairs = [], []
    all_pred_AC_types, all_true_AC_types = [], []
    if dataset_name == 'CDCP':
        orig_all_true_AR_pairs = [eval(rel_pairs) for rel_pairs in data_df["orig_AR_pairs"]]
        orig_all_true_AC_types_para = [eval(ac_types) for ac_types in data_df["orig_AC_types"]]
        orig_all_true_AC_types = []
        for __ in orig_all_true_AC_types_para:
            for _ in __:
                orig_all_true_AC_types.append(config.AC_type2id[_])
    all_pred_states, ACs_num_list = [], []

    for batch in data_loader:
        AC_bow_lists, AC_spans_list = batch['bow_vecs'], batch['AC_spans']
        AC_positions_list, AC_para_types_list = batch['AC_positions'], batch['AC_para_types']
        para_tokens_ids_list, parser_state_list = batch['para_tokens_ids'], batch['true_parser_states']
        true_AC_types_list, true_AR_pairs_list = batch['true_AC_types'], batch['true_AR_pairs']
        all_true_AR_pairs.extend(true_AR_pairs_list)

        for AC_types in true_AC_types_list:
            ACs_num_list.append(len(AC_types))
            all_true_AC_types += AC_types

        _, para_AC_reps_list = base_encoder(para_tokens_ids_list, AC_bow_lists,
                                            AC_spans_list, AC_positions_list, 
                                            AC_para_types_list)
        pred_pairs_list, pred_states_list, pred_AC_types = trans_model(para_AC_reps_list, 
                                                            parser_state_list, 'eval')
        all_pred_AR_pairs.extend(pred_pairs_list)
        all_pred_states.extend(pred_states_list)
        all_pred_AC_types.extend(pred_AC_types.tolist())
    
    if dataset_name == 'PE':
        eval_res = {"ARI-Macro": None, "Rel": None, "No-Rel": None,
            "ACTC-Macro": None, "MC": None, "Claim": None, "Premise": None,
            "Total": None}
        AC_type_metric = f1_score(all_true_AC_types, all_pred_AC_types, labels=[0, 1, 2], average=None)
        eval_res["MC"], eval_res["Claim"], eval_res["Premise"] = AC_type_metric
        eval_res['ACTC-Macro'] = (eval_res["MC"] + eval_res["Claim"] + eval_res["Premise"]) / 3
    elif dataset_name == 'CDCP':
        eval_res = {"ARI-Macro": None, "Rel": None, "No-Rel": None,
            "ACTC-Macro": None, "Value": None, "Testimony": None, "Policy": None, "Fact": None, "Reference": None,
            "Total": None}
        orig_all_pred_AC_types = align_with_orig(all_pred_AC_types, ACs_num_list, orig_all_true_AC_types_para)
        ACs_num_list = list(data_df["orig_adu_spans"].apply(lambda x: len(eval(x))))
        all_true_AR_pairs = orig_all_true_AR_pairs
        all_pred_AC_types = orig_all_pred_AC_types
        all_true_AC_types = orig_all_true_AC_types
        AC_type_metric = f1_score(all_true_AC_types, all_pred_AC_types, labels=[0, 1, 2, 3, 4], average=None)
        eval_res["Value"], eval_res["Testimony"], eval_res["Policy"], \
        eval_res["Fact"], eval_res["Reference"] = AC_type_metric
        eval_res["ACTC-Macro"] = (eval_res["Value"] + eval_res["Testimony"] \
                        + eval_res["Policy"] + eval_res["Fact"] + eval_res["Reference"]) / 5

    rel_metric = pair_metric(all_pred_AR_pairs, all_true_AR_pairs)
    eval_res["Rel"] = rel_metric[-1]
    all_true_no_rel_pairs = get_no_rel_pairs(all_true_AR_pairs, ACs_num_list)
    all_pred_no_rel_pairs = get_no_rel_pairs(all_pred_AR_pairs, ACs_num_list)
    no_rel_metric = pair_metric(all_pred_no_rel_pairs, all_true_no_rel_pairs)
    eval_res["No-Rel"] = no_rel_metric[-1]
    eval_res["ARI-Macro"] = (eval_res["Rel"] + eval_res["No-Rel"]) / 2

    if dataset_name == 'PE':
        eval_res["Total"] = (eval_res["ARI-Macro"] + eval_res["ACTC-Macro"]) / 2
    elif dataset_name == 'CDCP':
        eval_res["Total"] = (eval_res["Rel"] + eval_res["ACTC-Macro"]) / 2


    return eval_res, data_df

