# -*- coding: utf-8 -*-
# @Time        : 2023/3/16 23:59
# @Author      : ssxy00, lz
# @File        : gpt2_tokenizer.py
# @Description :


from transformers import GPT2Tokenizer

ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ("<bot>", "<human>", "<fact>")}
SPECIAL_TOKENS = ['<bos>', '<eos>', '<pad>', "<bot>", "<human>", "<fact>"]

class GPT2Vocab:

    def __init__(self, model_path):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.add_special_tokens_()


    def add_special_tokens_(self):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        orig_num_tokens = len(self.tokenizer.encoder)
        num_added_tokens = self.tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN)  # doesn't add if they are already there
        self.special_tokens = SPECIAL_TOKENS

    def __len__(self):
        return len(self.tokenizer)

    @property
    def bos_id(self):
        return self.tokenizer.bos_token_id

    @property
    def eos_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_id(self):
        return self.tokenizer.pad_token_id

    @property
    def bot_id(self):
        return self.tokenizer.convert_tokens_to_ids(["<bot>"])[0]

    @property
    def human_id(self):
        return self.tokenizer.convert_tokens_to_ids(["<human>"])[0]

    @property
    def fact_id(self):
        return self.tokenizer.convert_tokens_to_ids(["<fact>"])[0]


    def string2ids(self, string):
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(string, add_prefix_space=True))


    def ids2string(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=True):
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens,
                                     clean_up_tokenization_spaces=clean_up_tokenization_spaces)

