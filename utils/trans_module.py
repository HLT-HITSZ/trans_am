import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./utils')
from transformers import BertModel, BertTokenizer
from copy import deepcopy
import numpy as np
from copy import deepcopy
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

class BertEncoder(nn.Module):
    
    def __init__(self, vocab_dict, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.special_tokens = {'[PAD]', '<essay>', '<para-conclusion>', 
                                '<para-body>', '<para-intro>', '<ac>',
                                '</essay>', '</para-conclusion>', '</para-body>', 
                                '</para-intro>', '</ac>'}
        self.vocab_dict = vocab_dict
        self.bow_feature_size = len(self.vocab_dict)
        self.bow_rep_size = config.bert_output_size
        self.bow_fc_layer = nn.Linear(in_features=self.bow_feature_size,
                                    out_features=self.bow_rep_size)
        self.position_ebd_dim = config.position_ebd_dim
        self.max_AC_num = config.max_AC_num
        self.position_embedding = nn.Embedding(self.max_AC_num, self.position_ebd_dim)
        self.position_trainable = False
        self.para_type_embedding = nn.Embedding(self.max_AC_num, self.position_ebd_dim)
        self.para_type_trainable = False
        self.position_embedding.weight.requires_grad = self.position_trainable
        self.para_type_embedding.weight.requires_grad = self.para_type_trainable
        self.dropout = config.dropout
        self.dataset_name = config.dataset_name
    
    def padding_and_mask(self, ids_list):
        max_len = max([len(x) for x in ids_list])
        mask_list = []
        ids_padding_list = []
        for ids in ids_list:
            mask = [1.] * len(ids) + [0.] * (max_len - len(ids))
            ids = ids + [0] * (max_len - len(ids))
            mask_list.append(mask)
            ids_padding_list.append(ids)
        return ids_padding_list, mask_list
        
    def forward(self, para_tokens_ids_list, AC_bow_list, AC_spans_list, \
                    AC_positions_list, AC_para_types_list):
        ids_padding_list, mask_list = self.padding_and_mask(para_tokens_ids_list)
        ids_padding_tensor = torch.LongTensor(ids_padding_list).cuda()
        mask_tensor = torch.tensor(mask_list).cuda()
        bert_outputs = self.bert(ids_padding_tensor, attention_mask = mask_tensor)
        bert_emb = bert_outputs.last_hidden_state
        AC_bert_reps = []
        for batch_i, AC_spans in enumerate(AC_spans_list):
            for AC_span in AC_spans:
                AC_span_seq = bert_emb[batch_i][AC_span[0]:AC_span[1]+1]
                AC_bert_rep = torch.mean(AC_span_seq, 0)
                AC_bert_reps.append(AC_bert_rep)
        AC_bert_reps = torch.stack(AC_bert_reps)

        AC_bow_tensor = torch.cat(AC_bow_list, dim=0).cuda()
        AC_bow_reps = torch.sigmoid(self.bow_fc_layer(AC_bow_tensor))
        AC_bow_reps = F.dropout(AC_bow_reps, self.dropout, self.training)

        if self.dataset_name == 'PE':
            forward_pos_tensor = torch.tensor([x for xx in AC_positions_list for x in xx]).cuda()
            para_type_tensor = torch.tensor([x for xx in AC_para_types_list for x in xx]).cuda()
            forward_pos_emb = self.position_embedding(forward_pos_tensor)
            para_type_emb = self.para_type_embedding(para_type_tensor)

            AC_reps = torch.cat([AC_bert_reps, AC_bow_reps, forward_pos_emb, para_type_emb], dim=-1)
        elif self.dataset_name == 'CDCP':
            AC_para_types_list = [torch.tensor(_).cuda() + 2*self.max_AC_num for _ in AC_para_types_list]
            forward_pos_list = [torch.tensor(_).cuda() for _ in AC_positions_list]
            backward_pos_list = [reversed(self.max_AC_num + torch.tensor(_)).cuda() for _ in AC_positions_list]
            para_type_list = AC_para_types_list
            for_onehot_pos_list, back_onehot_pos_list, para_onehot_pos_list = \
                self.get_onehot_position_info(forward_pos_list, backward_pos_list, para_type_list)
            for_onehot_pos_tensor = torch.cat(for_onehot_pos_list, dim=0)
            back_onehot_pos_tensor = torch.cat(back_onehot_pos_list, dim=0)
            para_onehot_pos_tensor = torch.cat(para_onehot_pos_list, dim=0)
            onehot_pos_info =  torch.cat((for_onehot_pos_tensor, \
                                        back_onehot_pos_tensor, \
                                        para_onehot_pos_tensor), dim=-1)
            AC_reps = torch.cat([AC_bert_reps, onehot_pos_info, AC_bow_reps], dim=-1)

        AC_nums = [len(AC_spans) for AC_spans in AC_spans_list]
        para_AC_reps_list = torch.split(AC_reps, AC_nums, 0)
        return AC_reps, para_AC_reps_list

    def get_onehot_position_info(self, forward_pos_list, backward_pos_list, para_type_list):
        batch_size = len(forward_pos_list)

        def pos2onehot(pos, max_AC_num):
            pos = pos % max_AC_num
            pos_onehot = torch.zeros(pos.shape[0], max_AC_num)
            pos = pos.unsqueeze(-1)
            pos_onehot = pos_onehot.scatter_(1, pos.cpu(), 1).cuda()
            return pos_onehot

        for_onehot_pos_list = list(map(pos2onehot, forward_pos_list, [self.max_AC_num] * batch_size))
        back_onehot_pos_list = list(map(pos2onehot, backward_pos_list, [self.max_AC_num] * batch_size))
        para_onehot_pos_list = list(map(pos2onehot, para_type_list, [self.max_AC_num] * batch_size))
        return for_onehot_pos_list, back_onehot_pos_list, para_onehot_pos_list
 
class TransitionModel(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.is_bi = config.is_bi
        self.bert_output_size = config.bert_output_size
        self.mlp_size = config.mlp_size
        self.cell_size = config.cell_size
        self.scale_factor = config.scale_factor
        self.dropout = config.dropout
        self.lstm_layers = config.lstm_layers
        self.max_AC_num = config.max_AC_num
        self.max_dist_len = config.max_dist_len
        self.position_ebd_dim = config.position_ebd_dim

        self.distance_embedding = nn.Embedding(self.max_dist_len, self.position_ebd_dim)
        self.position_trainable = config.position_trainable
        self.action_ebd_dim = config.action_ebd_dim
        self.action_type_num = config.action_type_num
        self.action_embedding = nn.Embedding(self.action_type_num, self.action_ebd_dim) 
        self.action_trainable = config.action_trainable
        self.action_label_num = config.action_label_num
        self.AC_type_label_num = config.AC_type_label_num
        self.dataset_name = config.dataset_name
        if self.dataset_name == 'PE':
            self.AC_type_input_size = self.bert_output_size * 2 + self.position_ebd_dim * 2
            self.rel_type_input_size = self.bert_output_size * 2 + self.position_ebd_dim * 2
            self.lstm_input_size = self.bert_output_size * 2 + self.position_ebd_dim * 2
        elif self.dataset_name == 'CDCP':
            self.AC_type_input_size = self.bert_output_size * 2 + self.max_AC_num * 3
            self.lstm_input_size = self.bert_output_size * 2 + self.max_AC_num * 3

        self.stack_lstm = nn.LSTM(self.lstm_input_size, self.cell_size, self.lstm_layers, bidirectional=self.is_bi)
        self.buffer_lstm = nn.LSTM(self.lstm_input_size, self.cell_size, self.lstm_layers, bidirectional=self.is_bi)
        self.action_lstm = nn.LSTM(self.action_ebd_dim, self.cell_size, self.lstm_layers, bidirectional=False)
        self.trans_input_size = self.cell_size * 6 + self.position_ebd_dim + self.action_ebd_dim
        self.action_MLP = nn.Sequential(
            nn.Linear(self.trans_input_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size//self.scale_factor, self.action_label_num)
        )
        self.AC_type_mlp = nn.Sequential(
            nn.Linear(self.AC_type_input_size, self.mlp_size),
            nn.BatchNorm1d(self.mlp_size),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size, self.mlp_size//self.scale_factor),
            nn.BatchNorm1d(self.mlp_size//self.scale_factor),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.mlp_size//self.scale_factor, self.AC_type_label_num)
        )
        
        self.init_weight()
        
    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find("weight") != -1:
                if len(param.data.size()) > 1:
                    nn.init.xavier_normal_(param.data)
                else:
                    param.data.uniform_(-0.1, 0.1)
            elif name.find("bias") != -1:
                param.data.uniform_(-0.1, 0.1)
            else:
                continue
        self.distance_embedding.weight.requires_grad = self.position_trainable
        self.action_embedding.weight.requires_grad = self.action_trainable
    
    def action_encoder(self, parser_state_list):
        action_list = [[x[-1] for x in asl] for asl in parser_state_list]
        action_len_list = [len(x) for x in action_list]
        max_action_len = max(action_len_list)        
        action_padding_list = [[3]+x[:-1]+[4]*(max_action_len-len(x)) \
                        if len(x) != 0 else [3]+[4]*(max_action_len-1) \
                        for x in action_list]
        action_padding_tensor = torch.tensor(action_padding_list).cuda()
        
        inputs = self.action_embedding(action_padding_tensor).permute(1, 0, 2)
        bs = inputs.size()[1]
        outputs, _ = self.action_lstm(inputs)
        outputs_permute = outputs.permute(1, 0, 2)
        output_list = [outputs_permute[i][:al] for i, al in enumerate(action_len_list)]
        output_stack = torch.cat(output_list)
        return output_stack    

    
    def train_mode(self, para_AC_reps_list, parser_state_list):
        AC_reps = torch.cat(para_AC_reps_list, dim=0)
        AC_types_logits_tensor = self.AC_type_mlp(AC_reps)

        true_action_list, distance_list = [], []
        sk_reps_list, bf_reps_list = [], []

        for para_AC_reps, parser_state in zip(para_AC_reps_list, parser_state_list):
            for state in parser_state:
                sk, bf, delay_set, action = state[0], state[1], state[2], state[3]
                true_action_list.append(action)
                stack_reps = torch.stack([para_AC_reps[s] for s in sk])
                sk_reps_list.append(stack_reps)
                distance_list.append(int(abs(sk[1] - sk[0])))

                if len(bf) > 0:
                    bf_reps = torch.stack([para_AC_reps[b] for b in bf])
                else:
                    if sk[0] < para_AC_reps.size()[0]-1:
                        bf_reps = torch.stack([para_AC_reps[sk[0]+1]])
                    else:
                        bf_reps = torch.stack([para_AC_reps[sk[0]]])
                bf_reps_list.append(bf_reps)

        sk_reps_packed = pack_sequence(sk_reps_list, enforce_sorted=False)
        bf_reps_packed = pack_sequence(bf_reps_list, enforce_sorted=False)
        sk_lstm_out_packed, _ = self.stack_lstm(sk_reps_packed)
        bf_lstm_out_packed, _ = self.buffer_lstm(bf_reps_packed)
        sk_lstm_out_padded, sk_len_tensor = pad_packed_sequence(sk_lstm_out_packed, batch_first=True)
        bf_lstm_out_padded, bf_len_tensor = pad_packed_sequence(bf_lstm_out_packed, batch_first=True)
        sk_reps_list = [sk_lstm_out_padded[i][:sk_len] for i, sk_len in enumerate(sk_len_tensor)]
        bf_reps_list = [bf_lstm_out_padded[i][:bf_len] for i, bf_len in enumerate(bf_len_tensor)]
        hist_action_tensor = self.action_encoder(parser_state_list)

        state_reps_list = []
        for sk_reps, bf_reps in zip(sk_reps_list, bf_reps_list):
            state_reps = torch.cat([sk_reps[1], sk_reps[0], bf_reps[0]])
            state_reps_list.append(state_reps)

        distance_tensor = torch.tensor(distance_list).cuda()
        distance_embedding = self.distance_embedding(distance_tensor)
        final_feat_tensor = torch.cat([torch.stack(state_reps_list), 
                                            distance_embedding,
                                            hist_action_tensor], 1)
        true_action_tensor = torch.LongTensor(true_action_list).cuda()
        action_logits = self.action_MLP(final_feat_tensor)
        
        return action_logits, true_action_tensor, \
            AC_types_logits_tensor

    def predict_action(self, state, sk, bf, action, act_hidden=None): 
        sk_reps = torch.stack([state[s] for s in sk]).unsqueeze(0)
        if len(bf) > 0:
            bf_reps = torch.stack([state[b] for b in bf]).unsqueeze(0)
        else:
            if sk[0] < state.size()[0]-1:
                bf_reps = torch.stack([state[sk[0]+1]]).unsqueeze(0)
            else:
                bf_reps = torch.stack([state[sk[0]]]).unsqueeze(0)

        act_reps = self.action_embedding(torch.tensor([[action]]).cuda())

        sk_lstm_out, _ = self.stack_lstm(sk_reps.permute(1, 0, 2))
        sk_reps = sk_lstm_out.permute(1, 0, 2).squeeze(0)

        bf_lstm_out, _ = self.buffer_lstm(bf_reps.permute(1, 0, 2))
        bf_reps = bf_lstm_out.permute(1, 0, 2).squeeze(0)
        
        act_lstm_out, act_hidden = self.action_lstm(act_reps, act_hidden)
        act_reps = act_lstm_out.squeeze(0).squeeze(0)
        
        state_reps = torch.cat([sk_reps[1], sk_reps[0], bf_reps[0]])
        distance = torch.tensor(int(abs(sk[1] - sk[0]))).cuda()
        distance_emb = self.distance_embedding(distance)

        final_feat_tensor = torch.cat([state_reps, distance_emb, act_reps]).unsqueeze(0)

        action_logits = self.action_MLP(final_feat_tensor)
        action_probs = F.softmax(action_logits, 1)
        pred_action = action_probs.argmax(1).data.cpu().numpy()[0]

        return pred_action, act_hidden
    
    def eval_mode(self, para_AC_reps_list):
        id2action = {0: 'shift', 
                    1: 'right_arc_lt',
                    2: 'left_arc_lt'}
        pred_pairs_list = []
        pred_states_list = []
        for para_AC_reps in para_AC_reps_list:
            states = []
            if para_AC_reps.shape[0] == 1:
                pred_pairs = set()
                pred_pairs_list.append(pred_pairs)
                pred_states_list.append(states)
                continue

            pred_pairs, stack = [], []
            ACs_num = para_AC_reps.size()[0]
            buffer = list(range(ACs_num))
            delay_set = set()

            stack.insert(0, 0), stack.insert(0, 1)
            buffer.remove(0), buffer.remove(1)
            state = para_AC_reps

            action = 3
            act_hidden = None
            while len(buffer) > 0:
                if len(stack) < 2:
                    stack.insert(0, buffer.pop(0))
                s_pair = (stack[0], stack[1])
            
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)
                if id2action[action] == 'right_arc_lt':
                    s_pair = (stack[1], stack[0])
                states.append((deepcopy(stack), \
                                deepcopy(buffer), \
                                deepcopy(delay_set), \
                                action))
                if id2action[action] == 'shift':
                    if s_pair[1] in delay_set:  # DE_d
                        delay_set.remove(stack[1])
                        stack.pop(1)
                    elif len(buffer) > 0:   # SH
                        stack.insert(0, buffer.pop(0))
                        
                if id2action[action] == 'left_arc_lt':  # LA
                    stack.pop(1)
                    pred_pairs.append(s_pair)
                        
                if id2action[action] == 'right_arc_lt': # RA_d
                    pred_pairs.append(s_pair)
                    delay_set.add(s_pair[1])
                    if len(buffer) > 0:
                        stack.insert(0, buffer.pop(0))

            while len(stack) >= 2:
                s_pair = (stack[0], stack[1])
                action, act_hidden = self.predict_action(state, stack, buffer, action, act_hidden)
                if id2action[action] == 'right_arc_lt':
                    s_pair = (stack[1], stack[0])
                states.append((deepcopy(stack), \
                                deepcopy(buffer), \
                                deepcopy(delay_set), \
                                action))
                
                if id2action[action] == 'shift':    # DE
                    stack.pop(1)
                
                if id2action[action] == 'left_arc_lt':  # LA
                    stack.pop(1)
                    pred_pairs.append(s_pair)
                        
                if id2action[action] == 'right_arc_lt': # RA
                    pred_pairs.append(s_pair)
                    stack.pop(0)


            pred_pairs_list.append(set(pred_pairs))
            pred_states_list.append(states)
        
        AC_reps = torch.cat(para_AC_reps_list, dim=0)
        AC_types_logits_tensor = self.AC_type_mlp(AC_reps)
        pred_AC_types = torch.argmax(AC_types_logits_tensor, dim=-1)

        return pred_pairs_list, pred_states_list, pred_AC_types
    
    def forward(self, para_AC_reps_list, parser_state_list, mode):            
        if mode == 'train':
            action_logits, true_action_tensor, AC_types_logits_tensor = \
                self.train_mode(para_AC_reps_list, parser_state_list)
            return action_logits, true_action_tensor, AC_types_logits_tensor
        elif mode == 'eval':
            pred_pairs_list, pred_states_list, pred_AC_types = \
                self.eval_mode(para_AC_reps_list)
            return pred_pairs_list, pred_states_list, pred_AC_types
        else:
            print ('mode error!')
