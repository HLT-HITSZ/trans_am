import argparse

def get_config():
    config = argparse.ArgumentParser()
    config.add_argument("--dataset-name", default="CDCP", type=str)
    config.add_argument("--data-path", default="./data/cdcp_data_bert_df.csv", type=str)
    config.add_argument("--split-test-file-path", default="./data/cdcp_test_index.json", type=str)
    config.add_argument("--vocab-path", default="./data/bow_vocab_cdcp.json", type=str)
    config.add_argument("--save-path", default="./saved_models", type=str)
    config.add_argument("--bert-path", default="./data/bert-base-uncased", type=str)

    # training
    config.add_argument("--device", default='0', type=str)
    config.add_argument("--seed", default=0, type=int)
    config.add_argument("--batch-size", default=16, type=int)
    config.add_argument("--epochs", default=50, type=int)
    config.add_argument("--showtime", default=10, type=int)
    config.add_argument("--base-encoder-lr", default=1e-5, type=float)
    config.add_argument("--finetune-lr", default=1e-3, type=float)
    config.add_argument("--warm-up", default=5e-2, type=float)
    config.add_argument("--weight-decay", default=1e-5, type=float)
    config.add_argument("--early-num", default=10, type=int)

    # trans model param
    config.add_argument("--cell-size", default=256, type=int)
    config.add_argument("--lstm-layers", default=1, type=int)
    config.add_argument("--is-bi", default=True, type=bool)
    config.add_argument("--bert-output-size", default=768, type=int)
    config.add_argument("--mlp-size", default=512, type=int)
    config.add_argument("--scale-factor", default=2, type=int)
    config.add_argument("--max-AC-num", default=12, type=int)
    config.add_argument("--position-ebd-dim", default=256, type=int)
    config.add_argument("--action-ebd-dim", default=256, type=int)
    config.add_argument("--action-type-num", default=5, type=int)
    config.add_argument("--action-label-num", default=3, type=int)
    config.add_argument("--AC-type-label-num", default=5, type=int)
    config.add_argument("--position-trainable", default=True, type=bool)
    config.add_argument("--action-trainable", default=True, type=bool)
    config.add_argument("--dropout", default=0.5, type=float)
    config.add_argument("--max-dist-len", default=100, type=int)
    config.add_argument("--max-grad-norm", default=1.0, type=float)
    config = config.parse_args()

    if config.dataset_name == 'PE':
        config.AC_type2id = {"MajorClaim": 0, "Claim": 1, "Premise": 2}
    elif config.dataset_name == 'CDCP':
        config.AC_type2id = {"value": 0, "testimony": 1, "policy": 2, "fact": 3, "reference": 4}
    config.para_type2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    return config