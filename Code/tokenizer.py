from transformers import AdamW
from transformers import get_scheduler
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import AutoTokenizer

from tqdm import tqdm

import random
import evaluate

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import time

def get_training_corpus(dataset):
    corpus_list = []

    for start_idx in tqdm(range(0, len(dataset["train"]), 1000)):
        en_corpus = ''
        de_corpus = ''

        list_dict = dataset['train'].select(range(start_idx, min(start_idx + 1000, len(dataset['train']))))["translation"]

        for dict_trans in list_dict:
            en_corpus += dict_trans['en']
            en_corpus += ' '

            de_corpus += dict_trans['de']
            de_corpus += ' '

        corpus_list.append(en_corpus)
        corpus_list.append(de_corpus)

    return corpus_list

def tokenizer_spec_test(tokenizer, raw_datasets):
    metric = evaluate.load('sacrebleu') #spec metric: sacrebleu

    random.seed(0)

    rand_idx = random.sample(range(0, len(raw_datasets["train"])), 100)
    rand_samples = raw_datasets['train'].select(rand_idx)["translation"]

    en_sentences = []
    de_sentences = []

    en_decoded = []
    de_decoded = []

    cnt = 0
    for ex in tqdm(rand_samples):
        en_ex = ex["en"]
        de_ex = ex["de"]

        en_ex_encode = tokenizer.encode(en_ex)
        de_ex_encode = tokenizer.encode(de_ex)

        if type(en_ex_encode) == list:
            en_ex_decode = tokenizer.decode(en_ex_encode,
                                            skip_special_tokens = True,)
            de_ex_decode = tokenizer.decode(de_ex_encode,
                                            skip_special_tokens = True,)
        else:
            en_ex_decode = tokenizer.decode(en_ex_encode.ids,
                                            skip_special_tokens = True,)
            en_ex_decode = en_ex_decode.replace(' ##', '')
            de_ex_decode = tokenizer.decode(de_ex_encode.ids,
                                            skip_special_tokens = True,)
            de_ex_decode = de_ex_decode.replace(' ##', '')

        en_sentences.append(en_ex)
        de_sentences.append(de_ex)

        en_decoded.append(en_ex_decode)
        de_decoded.append(de_ex_decode)

        if cnt < 5:
            print("en_ex original setence: {}, en_ex decoded setence : {}".format(en_ex, en_ex_decode))
            print("de_ex original setence: {}, de_ex decoded setence : {}".format(de_ex, de_ex_decode))
            cnt += 1

    # prepare dataset for sacrebleu metric input
    en_pred = en_decoded
    en_ref = en_sentences

    de_pred = de_decoded
    de_ref = de_sentences

    en_metric = metric.compute(predictions = en_pred, references = en_ref)
    de_metric = metric.compute(predictions = de_pred, references = de_ref)

    print("*"*10)

    if 'get_vocab_size' in dir(tokenizer):
        print("tokenizer vocab size: {}".format(tokenizer.get_vocab_size()))
    elif 'vocab_size' in dir(tokenizer):
        print("tokenizer vocab size: {}".format(tokenizer.vocab_size))
    print("english sacrebleu scores: {}".format(en_metric))
    print("german sacrebleu scores: {}".format(de_metric))

    return en_decoded, de_decoded

def tokenizer_spec_test_using_unseen(tokenizer, raw_datasets):
    metric = evaluate.load('sacrebleu') #spec metric: sacrebleu

    random.seed(0)

    rand_idx = random.sample(range(0, len(raw_datasets["test"])), 1000)
    rand_samples = raw_datasets['test'].select(rand_idx)["translation"]

    en_sentences = []
    de_sentences = []

    en_decoded = []
    de_decoded = []

    cnt = 0
    for ex in tqdm(rand_samples):
        en_ex = ex["en"]
        de_ex = ex["de"]

        en_ex_encode = tokenizer.encode(en_ex)
        de_ex_encode = tokenizer.encode(de_ex)

        if type(en_ex_encode) == list:
            en_ex_decode = tokenizer.decode(en_ex_encode,
                                            skip_special_tokens = True,)
            de_ex_decode = tokenizer.decode(de_ex_encode,
                                            skip_special_tokens = True,)
        else:
            en_ex_decode = tokenizer.decode(en_ex_encode.ids,
                                            skip_special_tokens = True,)
            en_ex_decode = en_ex_decode.replace(' ##', '')
            de_ex_decode = tokenizer.decode(de_ex_encode.ids,
                                            skip_special_tokens = True,)
            de_ex_decode = de_ex_decode.replace(' ##', '')

        en_sentences.append(en_ex)
        de_sentences.append(de_ex)

        en_decoded.append(en_ex_decode)
        de_decoded.append(de_ex_decode)

        if cnt < 5:
            print("en_ex original setence: {}, en_ex decoded setence : {}".format(en_ex, en_ex_decode))
            print("de_ex original setence: {}, de_ex decoded setence : {}".format(de_ex, de_ex_decode))
            cnt += 1

    # prepare dataset for sacrebleu metric input
    en_pred = en_decoded
    en_ref = en_sentences

    de_pred = de_decoded
    de_ref = de_sentences

    en_metric = metric.compute(predictions = en_pred, references = en_ref)
    de_metric = metric.compute(predictions = de_pred, references = de_ref)

    print("*"*10)

    if 'get_vocab_size' in dir(tokenizer):
        print("tokenizer vocab size: {}".format(tokenizer.get_vocab_size()))
    elif 'vocab_size' in dir(tokenizer):
        print("tokenizer vocab size: {}".format(tokenizer.vocab_size))
    print("english sacrebleu scores: {}".format(en_metric))
    print("german sacrebleu scores: {}".format(de_metric))

    return en_decoded, de_decoded

if __name__ == '__main__':
    # laod datasets (wmt16, de-en)
    raw_datasets = load_dataset('wmt16', 'de-en')

    # get training corpus
    train_corpus = get_training_corpus(raw_datasets)

    # defining tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]", continuing_subword_prefix="##"))
    trainer = BpeTrainer(vocab_size = 20000,
                        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"],
                        continuing_subword_prefix = "##")
    tokenizer.pre_tokenizer = Whitespace()

    # training tokenizer
    start = time.time()
    tokenizer.train_from_iterator(
        train_corpus,
        trainer = trainer,
    )
    end = time.time()

    print(f"{end - start:.5f} sec")

    # check specs
    _, _ = tokenizer_spec_test(tokenizer, raw_datasets) # spec on training dataset
    _, _ = tokenizer_spec_test_using_unseen(tokenizer, raw_datasets) # spec on unseen dataset

    # save trained tokenizer
    tokenizer.save('./WMT_16_TOKENIZER/wmt_16_30000.json')

'''
How to load saved toknenizer?

file_dict = {
    'tokenizer_30000': './WMT_16_TOKENIZER/wmt_16_30000.json',
    'tokenizer_20000': './WMT_16_TOKENIZER/wmt_16_20000.json',
    'tokenizer_10000': './WMT_16_TOKENIZER/wmt_16_10000.json',
}

with open(file_dict['tokenizer_20000'], 'r') as f:
    tokenizer = Tokenizer.from_str(''.join(f.readlines()))
'''