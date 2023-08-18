from model import *

import copy
from tqdm.auto import tqdm
import os

from typing import Optional, Any, Union, Callable
import torch
from torch.nn import functional as F
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader

from transformers import get_scheduler
from transformers import get_inverse_sqrt_schedule

from datasets import load_dataset
from datasets import Dataset
from datasets import concatenate_datasets
import datasets

import pickle
import random

from tokenizers import Tokenizer
from tokenizers.models import BPE

# define data_collator
def data_collate_fn(samples):
    collate_en = []
    collate_de = []
    
    pad_idx = 3 # tokenizer.token_to_id('[PAD]')

    max_len_en = max([len(sample['en']) for sample in samples])
    max_len_de = max([len(sample['de']) for sample in samples])

    for sample in samples:
        diff_en = max_len_en - len(sample['en'])
        diff_de = max_len_de - len(sample['de'])

        if diff_en > 0:
            pad_tensor_en = torch.ones(size = (diff_en, )) * pad_idx
            collate_en.append(torch.cat([torch.tensor(sample['en']), pad_tensor_en], dim = 0))
        else:
            collate_en.append(torch.tensor(sample['en']))
        
        if diff_de > 0:
            pad_tensor_de = torch.ones(size = (diff_de, )) * pad_idx
            collate_de.append(torch.cat([torch.tensor(sample['de']), pad_tensor_de], dim = 0))
        else:
            collate_de.append(torch.tensor(sample['de']))

    return {'en': torch.stack(collate_en).type(torch.LongTensor), 'de': torch.stack(collate_de).type(torch.LongTensor)}

if __name__ == "__main__":
    random.seed(0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.autograd.set_detect_anomaly(True)

    #####
    # Part1. Preparing the data
    #####

    # parameters
    max_length = 500
    
    # load trained tokenizer
    # load tokenizer
    file_dict = {
        'tokenizer_30000': './WMT_16_TOKENIZER/wmt_16_30000.json',
        'tokenizer_20000': './WMT_16_TOKENIZER/wmt_16_20000.json',
        'tokenizer_10000': './WMT_16_TOKENIZER/wmt_16_10000.json',
    }

    with open(file_dict['tokenizer_10000'], 'r') as f:
        tokenizer = Tokenizer.from_str(''.join(f.readlines()))

    # load preprocessed datasets
    train_path_prefix = './WMT_16_PREPROCESSED/train/'
    
    cur_dir = os.getcwd()
    os.chdir(train_path_prefix)
    train_fs = os.listdir()
    os.chdir(cur_dir)
    
    valid_path_prefix = './WMT_16_PREPROCESSED/validation/'
    valid_fs = ['data-00000-of-00001.arrow', 
                'state.json', 
                'dataset_info.json']
    
    test_path_prefix = './WMT_16_PREPROCESSED/test/'
    test_fs = ['data-00000-of-00001.arrow', 
               'state.json', 
               'dataset_info.json']
    
    ## load train datasets
    train_datasets = []

    for f in train_fs:
        if f.split('.')[-1] == 'arrow':
            f_path = train_path_prefix + f
            
            temp_ds = Dataset.from_file(f_path)
            train_datasets.append(temp_ds)

    train_ds = concatenate_datasets(train_datasets)

    ## load valid datasets
    valid_datasets = []

    for f in valid_fs:
        if f.split('.')[-1] == 'arrow':
            f_path = valid_path_prefix + f
            
            temp_ds = Dataset.from_file(f_path)
            valid_datasets.append(temp_ds)

    valid_ds = concatenate_datasets(valid_datasets)

    ## load test datasets
    test_datasets = []

    for f in test_fs:
        if f.split('.')[-1] == 'arrow':
            f_path = test_path_prefix + f
            
            temp_ds = Dataset.from_file(f_path)
            test_datasets.append(temp_ds)

    test_ds = concatenate_datasets(test_datasets)

    # tokenized dataset
    '''
    DatasetDict({
        train: Dataset({
            features: ['en', 'de'],
            num_rows: 4548809
        })
        validation: Dataset({
            features: ['en', 'de'],
            num_rows: 2169
        })
        test: Dataset({
            features: ['en', 'de'],
            num_rows: 2999
        })
    })
    '''
    tok_datasets = datasets.DatasetDict({"train":train_ds, "validation": valid_ds,"test":test_ds})

    #####
    # Part2. Define Model
    #####

    # parameters
    d_model = 512
    nhead = 8
    d_ff = 2048
    dropout_rate = 0.1
    batch_first = True
    n_layers = 6
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id('[PAD]')
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # define language1 self attetntion 6ea
    for i in range(1, n_layers + 1):
        globals()['lan1_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout_rate)

    # define language2 self attention 6ea
    for i in range(1, n_layers + 1):
        globals()['lan2_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout_rate)

    # define language1, language2 multihead attention 6ea
    for i in range(1, n_layers + 1):
        globals()['lan1_lan2_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout_rate)

    # define encoderlayer for lan1 -> lan2
    for i in range(1, n_layers + 1):
        kwargs = {'self_attn': globals()['lan1_self_attention' + str(i)]}
        globals()['lan1_encoder_layer' + str(i)] = EncoderLayer(d_model, nhead, d_ff, dropout_rate, **kwargs)

    # define decoderlayer for lan1 -> lan2
    for i in range(1, n_layers + 1):
        kwargs = {
            'attn_layer': globals()['lan2_self_attention' + str(i)], 
            'enc_attn_layer': globals()['lan1_lan2_self_attention' + str(i)]
            }
        globals()['lan1_decoder_layer' + str(i)] = DecoderLayer(d_model, nhead, d_ff, dropout_rate, **kwargs)
            
    tok_embedding = nn.Embedding(vocab_size, d_model, 
                                        padding_idx=pad_idx)

    # encoder_layers for transformer1
    enc_list_for_trans1 = []
    for i in range(1, n_layers + 1):
        enc_list_for_trans1.append(globals()['lan1_encoder_layer' + str(i)])

    enc_layers_for_trans1 = nn.ModuleList(
        enc_list_for_trans1
    )

    kwargs_encoder_trans1 = {
        'tok_embedding': tok_embedding, 
        'layers': enc_layers_for_trans1
    }

    customized_encoder_for_trans1 = Encoder(vocab_size, d_model, n_layers=6, n_heads=nhead, d_ff=d_ff,
                                            pad_idx=pad_idx, dropout_rate=dropout_rate, max_len=max_length,
                                            **kwargs_encoder_trans1)

    # decoder_layers for transformer1
    dec_list_for_trans1 = []
    for i in range(1, n_layers + 1):
        dec_list_for_trans1.append(globals()['lan1_decoder_layer' + str(i)])

    dec_layers_for_trans1 = nn.ModuleList(
        dec_list_for_trans1
    )

    kwargs_decoder_trans1 = {
        'tok_embedding': tok_embedding,
        'layers': dec_layers_for_trans1
    }

    customized_decoder_for_trans1 = Decoder(vocab_size, d_model, n_layers=6, n_heads=nhead, d_ff=d_ff,
                                            pad_idx=pad_idx, dropout_rate=dropout_rate, max_len=max_length,
                                            **kwargs_decoder_trans1)

    generator_for_trans1 = Generator(d_model=d_model, vocab_size=vocab_size)

    model = Transformer(encoder=customized_encoder_for_trans1, decoder=customized_decoder_for_trans1,
                                            generator=generator_for_trans1, pad_idx=pad_idx)


    #####
    # Part3. Define DataLoader
    #####
    train_dataloader = DataLoader(
        tok_datasets["train"],
        shuffle = True,
        batch_size = 64,
        num_workers=4,
        collate_fn = data_collate_fn,
    )

    eval_dataloader = DataLoader(
        tok_datasets["validation"],
        batch_size = 64,
        num_workers=4,
        collate_fn = data_collate_fn,
    )

    #####
    # Part4. Design Training Loop
    #####
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, betas = (0.9, 0.98))

    # lr scheduler
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_inverse_sqrt_schedule(
        optimizer=optimizer,
        num_warmup_steps=4000,
    )

    # lr_scheduler = get_scheduler(
    #     "linear",
    #     optimizer = optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=num_training_steps
    # )

    with torch.autograd.detect_anomaly():
        print("before strat training epoch, confirm")

    # model to device
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    cross_entropy = nn.CrossEntropyLoss(ignore_index = pad_idx)

    train_loss_history = []
    valid_loss_history = []

    for epoch in range(num_epochs):
        loss = []
        loss_period = []
        min_loss = np.inf
        
        # model training
        model.train()
        cnt = 0

        for b in train_dataloader:
            batch = {k: v.to(device) for k, v in b.items()}
            
            lan1_ = batch['en'] # (batch, lan1_seq_len)
            lan2_ = batch['de'] # (batch, lan2_seq_len)
            
            lan1_decoder_output = lan1_[:, 1:].contiguous().view(-1)
            lan2_decoder_output = lan2_[:, 1:].contiguous().view(-1)

            ########
            # start from language 1.
            ########

            # act1. lan1 -> lan2 (loop_cond = 'start1_from_lan1_to_lan2')
            s1_lan1_lan2 = model(lan1_, lan2_)
            s1_lan1_lan2_1d = s1_lan1_lan2.contiguous().view(-1, vocab_size)
            s1_lan1_lan2_loss = cross_entropy(s1_lan1_lan2_1d, lan2_decoder_output)

            s1_lan1_lan2_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()

            progress_bar.update(1)

            loss_mean = s1_lan1_lan2_loss.item()

            loss.append(loss_mean)
            loss_period.append(loss_mean)
            
            del batch

            if cnt % 100 == 0:
                # valid loss
                valid_loss = []

                model.eval()

                for v_b in eval_dataloader:
                    v_batch = {k: v.to(device) for k, v in v_b.items()}

                    lan1_ = v_batch['en']
                    lan2_ = v_batch['de']

                    lan1_decoder_output = lan1_[:, 1:].contiguous().view(-1)
                    lan2_decoder_output = lan2_[:, 1:].contiguous().view(-1)

                    ########
                    # start from language 1.
                    ########

                    # act1. lan1 -> lan2 (loop_cond = 'start1_from_lan1_to_lan2')
                    s1_lan1_lan2 = model(lan1_, lan2_)
                    s1_lan1_lan2_1d = s1_lan1_lan2.contiguous().view(-1, vocab_size)
                    s1_lan1_lan2_loss = cross_entropy(s1_lan1_lan2_1d, lan2_decoder_output)

                    loss_mean = s1_lan1_lan2_loss.item()

                    loss.append(loss_mean)
                    valid_loss.append(loss_mean)

                    del v_batch

                # Loss Reporting
                print("epoch: {}, cnt : {}, train loss: {}, valid loss: {}".format(epoch, cnt, np.mean(loss_period), np.mean(valid_loss)))
                train_loss_history.append(np.mean(loss_period))
                valid_loss_history.append(np.mean(valid_loss))
                
                # get back to training mode
                model.train()

                # free loss_period
                loss_period = []
            
            cnt += 1