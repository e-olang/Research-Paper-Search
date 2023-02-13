from tokenizers import BertWordPieceTokenizer
import os
import json


special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
files = ["train.txt"]
vocab_size = 30_000

tokenizer = BertWordPieceTokenizer()
tokenizer.train(files=files, vocab_size=vocab_size,
                special_tokens=special_tokens)

tokenizer_path_30k = "WordPiece"


def write_tokenizer(tokenizer_path):
    # make the directory if not already there
    if not os.path.isdir(tokenizer_path):
        os.mkdir(tokenizer_path)

    # save the tokenizer
    tokenizer.save_model(tokenizer_path)

    # dumping some of the tokenizer config to config file,
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(tokenizer_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True,
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        }
        json.dump(tokenizer_cfg, f)


write_tokenizer(tokenizer_path_30k)
