import copy

action2id = {'shift': 0, 'right_arc_lt': 1,
             'left_arc_lt': 2}

def get_action(s_pair, pairs):
    action = None
    
    if s_pair in pairs:
        action = 'left_arc_lt'
    elif (s_pair[1], s_pair[0]) in pairs:
        action = 'right_arc_lt'
    else:
        action = 'shift'
    
    return action
        
def get_action_sequence(adu_spans, ac_rel_pairs, action2id):
    retrieved_pairs = []
    stack = []
    delay_delete = []
    target_pairs = ac_rel_pairs
    buffer = list(range(0, len(adu_spans)))
    stack.insert(0, 0), stack.insert(0, 1)
    buffer.remove(0), buffer.remove(1)
    actions = []
    while len(buffer) > 0:
        if len(stack) < 2:
            stack.insert(0, buffer.pop(0))
        s_pair = (stack[0], stack[1])
        
        action = get_action(s_pair, target_pairs)
        if action == 'right_arc_lt':
            s_pair = (stack[1], stack[0])
        actions.append((copy.deepcopy(stack), \
                        copy.deepcopy(buffer), \
                        copy.deepcopy(delay_delete), \
                        action2id[action]))
        if action == 'shift':
            if s_pair[1] in delay_delete: # DE_d
                delay_delete.remove(stack[1])
                stack.pop(1)
            elif len(buffer) > 0: # SH
                stack.insert(0, buffer.pop(0))
                
        if action == 'left_arc_lt': # LA
            stack.pop(1)
            retrieved_pairs.append(s_pair)
                
        if action == 'right_arc_lt': # RA_d
            retrieved_pairs.append(s_pair)
            delay_delete.append(s_pair[1])
            if len(buffer) > 0:
                stack.insert(0, buffer.pop(0))
    
    while len(stack) >= 2:
        s_pair = (stack[0], stack[1])
        action = get_action(s_pair, target_pairs)
        if action == 'right_arc_lt':
            s_pair = (stack[1], stack[0])
        actions.append((copy.deepcopy(stack), \
                        copy.deepcopy(buffer), \
                        copy.deepcopy(delay_delete), \
                        action2id[action]))
        
        if action == 'shift': # DE
            stack.pop(1)
        
        if action == 'left_arc_lt': # LA
            stack.pop(1)
            retrieved_pairs.append(s_pair)
                
        if action == 'right_arc_lt': # RA
            retrieved_pairs.append(s_pair)
            stack.pop(0)

    return actions # , set(retrieved_pairs)

def text_2_action_sequence(data_df, dataset_name, mode):
    action_sequence = []
    if dataset_name == 'PE':
        adu_spans_list = list(data_df["adu_spans"])
        ac_rel_pairs = list(data_df["AR_pairs"])
    elif dataset_name == 'CDCP':
        if mode == 'train':
            adu_spans_list = list(data_df["adu_spans"])
            ac_rel_pairs = list(data_df["AR_pairs"])
        else:
            adu_spans_list = list(data_df["orig_adu_spans"])
            ac_rel_pairs = list(data_df["orig_AR_pairs"])

    for adu_spans, ac_rel_pairs in zip(adu_spans_list, ac_rel_pairs):
        adu_spans = eval(adu_spans)
        ac_rel_pairs = eval(ac_rel_pairs)
        if len(adu_spans) >= 2:
            action = get_action_sequence(adu_spans, ac_rel_pairs, action2id)
        else:
            action = []
        action_sequence.append(action)
    return action_sequence
