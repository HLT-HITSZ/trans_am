import torch, os, time, random, sys, json
import numpy as np
import logging
import torch.nn as nn
sys.path.append('./utils')
from trans_module import TransitionModel, BertEncoder
from transformers import AdamW, get_linear_schedule_with_warmup
from evaluation import evaluate, get_eval_result
import datetime
from dataloader import ArgMiningDataset, generate_batch_fn
from torch.utils.data import DataLoader
import pandas as pd
from config import get_config
from tqdm import tqdm
config = get_config()
os.environ['CUDA_VISIBLE_DEVICES'] = config.device

now = datetime.datetime.now()
now_time_string = "{:0>4d}{:0>2d}{:0>2d}_{:0>2d}{:0>2d}{:0>2d}_{:0>5d}".format(
                now.year, now.month, now.day, now.hour, now.minute, now.second, config.seed)

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ['PYTHONHASHSEED'] = str(config.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

save_path = config.save_path
save_path = os.path.join(save_path, now_time_string)
if not os.path.exists(save_path):
    os.makedirs(save_path)
else:
    print("save_path exists!!")
    exit(1)
with open(os.path.join(save_path, "config.json"), "w") as fp:
    json.dump(config.__dict__, fp)

codes_save_path = os.path.join(save_path, 'codes')
os.makedirs(codes_save_path)
base_dir = os.getcwd()
for name in os.listdir(base_dir):
    if name == 'utils':
        os.system(f'cp -r {name} {codes_save_path}')
    elif name == 'preprocess_cdcp':
        os.system(f'cp -r {name} {codes_save_path}')
    elif not os.path.isdir(os.path.join(base_dir, name)):
        os.system(f'cp {name} {codes_save_path}')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch) # output to terminal
logger.addHandler(fh) # output to file

logger.info('Loading data...')
with open(config.vocab_path, 'r') as fp:
    vocab_dict = json.load(fp)

data_df = pd.read_csv(config.data_path)
with open(config.split_test_file_path, "r") as fp:
    test_id_list = json.load(fp)
test_data_df = data_df[data_df["para_id"].isin(test_id_list)]
train_data_df = data_df[~(data_df["para_id"].isin(test_id_list))]

if config.dataset_name == 'PE':
    essay_id2parag_id_dict = train_data_df.groupby("essay_id").groups
    essay_id_list = list(essay_id2parag_id_dict.keys())
    random.shuffle(essay_id_list)
    num_train_essay = int(len(essay_id_list) * 0.9)
    dev_essay_id_list = essay_id_list[num_train_essay:]
    dev_para_id_list = []
    for essay_id in dev_essay_id_list:
        dev_para_id_list += essay_id2parag_id_dict[essay_id].tolist()
elif config.dataset_name == 'CDCP':
    num_train_essay = int(len(train_data_df) * 0.85)
    essay_id_list = list(train_data_df["para_id"])
    random.shuffle(essay_id_list)
    dev_para_id_list = essay_id_list[num_train_essay:]
else:
    print('Wrong dataset name.')
    exit()

dev_data_df = train_data_df[train_data_df["para_id"].isin(dev_para_id_list)]
train_data_df = train_data_df[~train_data_df["para_id"].isin(dev_para_id_list)]

train_data_df = train_data_df[train_data_df["adu_spans"].apply(lambda x: len(eval(x))>0)]
dev_data_df = dev_data_df[dev_data_df["adu_spans"].apply(lambda x: len(eval(x))>0)]
test_data_df = test_data_df[test_data_df["adu_spans"].apply(lambda x: len(eval(x))>0)]

train_dataset = ArgMiningDataset(train_data_df, vocab_dict, config.dataset_name, 'train')
dev_dataset = ArgMiningDataset(dev_data_df, vocab_dict, config.dataset_name, 'dev')
test_dataset = ArgMiningDataset(test_data_df, vocab_dict, config.dataset_name, 'test')


train_len = len(train_dataset)
train_iter_len = (train_len // config.batch_size) + 1
if train_len % config.batch_size == 1:
    train_iter_len -= 1
num_training_steps = train_iter_len * config.epochs
num_warmup_steps = int(num_training_steps * config.warm_up)
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                            shuffle=True, collate_fn=generate_batch_fn)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, 
                            shuffle=False, collate_fn=generate_batch_fn)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, collate_fn=generate_batch_fn)
logger.info('Data loaded.')

logger.info('Initializing model...')
base_encoder = BertEncoder(vocab_dict, config)
base_encoder.cuda()
trans_model = TransitionModel(config)
trans_model.cuda()
logger.info('Model initialized.')

crossentropy = nn.CrossEntropyLoss()
base_encoder_optimizer = filter(lambda x: x.requires_grad, list(base_encoder.parameters()))
trans_optimizer = filter(lambda x: x.requires_grad, list(trans_model.parameters()))
optimizer_parameters = [
        {'params': [p for p in trans_optimizer if len(p.data.size()) > 1], 'weight_decay': config.weight_decay},
        {'params': [p for p in trans_optimizer if len(p.data.size()) == 1], 'weight_decay': 0.0},
        {'params': base_encoder_optimizer, 'lr': config.base_encoder_lr},
        {'params': trans_optimizer}
        ]

optimizer = AdamW(optimizer_parameters, config.finetune_lr)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                            num_training_steps=num_training_steps)

total_batch, early_stop = 0, 0
best_batch, best_f1 = 0, 0.0
for epoch_i in range(config.epochs):
    logger.info("Running epoch: {}".format(epoch_i))
    batch_i = 0
    for batch in tqdm(train_loader):
        batch_i += 1
        trans_model.train()
        base_encoder.train()
        optimizer.zero_grad()
        para_tokens_ids_list = batch['para_tokens_ids']
        AC_bow_lists, AC_spans_list = batch['bow_vecs'], batch['AC_spans']
        AC_positions_list, AC_para_types_list = batch['AC_positions'], batch['AC_para_types']
        parser_state_list, true_AC_types_list = batch['true_parser_states'], batch['true_AC_types']

        flat_true_AC_types_list = [_ for AC_types in true_AC_types_list for _ in AC_types]
        true_AC_types_tensor = torch.tensor(flat_true_AC_types_list).cuda()
        flat_parser_states = [_ for parser_states in parser_state_list for _ in parser_states]

        if true_AC_types_tensor.shape[0] <= 1 or len(flat_parser_states) <= 1:
            continue

        AC_reps, para_AC_reps_list = base_encoder(para_tokens_ids_list, AC_bow_lists,
                                                AC_spans_list, AC_positions_list,
                                                AC_para_types_list)
        action_logits, true_action_tensor, AC_types_logits = trans_model(para_AC_reps_list, 
                                            parser_state_list, 'train')

        action_loss = crossentropy(action_logits, true_action_tensor)
        AC_type_loss = crossentropy(AC_types_logits, true_AC_types_tensor)
        loss = action_loss + AC_type_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_batch += 1

        if total_batch % config.showtime == 0:
            t_start = time.time()
            dev_eval_res, dev_res_df = evaluate(trans_model, base_encoder, 
                                            dev_data_df, dev_loader, config.dataset_name, 'eval')
            t_end = time.time()
            if dev_eval_res["Total"] > best_f1:
                early_stop = 0
                best_f1 = dev_eval_res["Total"]
                best_batch = total_batch                    
                logger.info('*'*20 +'The performance on dev set' + '*'*20)
                logger.info('Eval running time: {}'.format(t_end - t_start))
                logger.info('Batch num: {}'.format(total_batch))
                ARI_msg, ACTC_msg, macro_msg = get_eval_result(dev_eval_res, config.dataset_name) 
                logger.info(ARI_msg)
                logger.info(ACTC_msg)
                logger.info(macro_msg)
                torch.save(base_encoder.state_dict(), os.path.join(save_path, 'bert_best.mdl'))
                torch.save(trans_model.state_dict(), os.path.join(save_path, 'trans_best.mdl'))

                logger.info('*'*20 +'The performance on test set' + '*'*20)
                test_eval_res, test_res_df = evaluate(trans_model, base_encoder, test_data_df, 
                                                            test_loader, config.dataset_name, "test")
                ARI_msg, ACTC_msg, macro_msg = get_eval_result(test_eval_res, config.dataset_name)
                logger.info(ARI_msg)
                logger.info(ACTC_msg)
                logger.info(macro_msg)     
    early_stop += 1
    if early_stop > config.early_num:
        logger.info('Early stop')
        break

# test set results
base_encoder.load_state_dict(torch.load(os.path.join(save_path, 'bert_best.mdl')))
trans_model.load_state_dict(torch.load(os.path.join(save_path, 'trans_best.mdl')))
logger.info('='*20 +'The performance on test set' + '='*20)
test_eval_res, test_res_df = evaluate(trans_model, base_encoder, 
                                    test_data_df, test_loader, config.dataset_name, "test")
logger.info('Best batch num: {}'.format(best_batch))
ARI_msg, ACTC_msg, macro_msg = get_eval_result(test_eval_res, config.dataset_name)
logger.info(ARI_msg)
logger.info(ACTC_msg)
logger.info(macro_msg)
rel_f1 = test_eval_res["Rel"]
with open(os.path.join(save_path, 'result.txt'), 'w') as fp:
    fp.write(ARI_msg + '\n')
    fp.write(ACTC_msg + '\n')
    fp.write(macro_msg + '\n')
test_res_df.to_csv(os.path.join(save_path, "test_pred_res.csv"), index=False)
