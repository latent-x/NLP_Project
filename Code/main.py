from model import *

import copy
from tqdm.auto import tqdm
import os

from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from torch.nn import MultiheadAttention
from torch.nn import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import LayerNorm
from torch.nn import MultiheadAttention
import torch.nn as nn
import numpy as np
import math
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import get_scheduler
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer

from datasets import load_dataset
from datasets import Dataset
from datasets import concatenate_datasets
import datasets
import pickle

from tokenizers import Tokenizer
from tokenizers.models import BPE

from torch.utils.tensorboard import SummaryWriter

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.autograd.set_detect_anomaly(True)

    #####
    # Part1. Preparing the data
    #####

    # parameters
    max_length = 5000
    
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
    train_fs = ['data-00000-of-00006.arrow',
                'data-00001-of-00006.arrow',
                'data-00002-of-00006.arrow',
                'data-00003-of-00006.arrow',
                'data-00004-of-00006.arrow',
                'data-00005-of-00006.arrow',
                'state.json',
                'dataset_info.json']
    
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
    dropout_attention = 0.0
    batch_first = True
    n_layers = 6
    vocab_size = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id('[PAD]')
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    # define language1 self attetntion 6ea
    for i in range(1, n_layers + 1):
        globals()['lan1_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout=dropout_attention, 
                                                                    batch_first=batch_first, device = device)

    # define language2 self attention 6ea
    for i in range(1, n_layers + 1):
        globals()['lan2_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout=dropout_attention, 
                                                                    batch_first=batch_first, device = device)

    # define language1, language2 multihead attention 6ea
    for i in range(1, n_layers + 1):
        globals()['lan1_lan2_self_attention' + str(i)] = MultiheadAttention(d_model, nhead, dropout=dropout_attention, 
                                                                    batch_first=batch_first, device = device)

    # define encoderlayer for lan1 -> lan2
    for i in range(1, n_layers + 1):
        kwargs = {'self_attn': globals()['lan1_self_attention' + str(i)]}
        globals()['lan1_encoder_layer' + str(i)] = ProposedEncoderLayer(d_model = d_model, nhead = nhead, 
                                                                        device = device, batch_first = batch_first, **kwargs)

    # define encoderlayer for lan2 -> lan1
    for i in range(1, n_layers + 1):
        kwargs = {'self_attn': globals()['lan2_self_attention' + str(i)]}
        globals()['lan2_encoder_layer' + str(i)] = ProposedEncoderLayer(d_model = d_model, nhead = nhead, 
                                                                        device = device, batch_first = batch_first, **kwargs)

    # define decoderlayer for lan1 -> lan2
    for i in range(1, n_layers + 1):
        kwargs = {
            'self_attn': globals()['lan2_self_attention' + str(i)], 
            'multihead_attn': globals()['lan1_lan2_self_attention' + str(i)]
            }
        globals()['lan1_decoder_layer' + str(i)] = ProposedDecoderLayer(d_model = d_model, nhead = nhead, 
                                                                        device = device, batch_first = batch_first, **kwargs)
        
    # define decoderlayer for lan2 -> lan1
    for i in range(1, n_layers + 1):
        kwargs = {
            'self_attn': globals()['lan1_self_attention' + str(i)], 
            'multihead_attn': globals()['lan1_lan2_self_attention' + str(i)]
            }
        globals()['lan2_decoder_layer' + str(i)] = ProposedDecoderLayer(d_model = d_model, nhead = nhead, 
                                                                        device = device, batch_first = batch_first, **kwargs)
        
    # encoder_layers for transformer1
    enc_list_for_trans1 = []
    for i in range(1, n_layers + 1):
        enc_list_for_trans1.append(globals()['lan1_encoder_layer' + str(i)])

    enc_layers_for_trans1 = nn.ModuleList(
        enc_list_for_trans1
    )

    customized_encoder_for_trans1 = ProposedEncoder(enc_layers_for_trans1)

    # decoder_layers for transformer1
    dec_list_for_trans1 = []
    for i in range(1, n_layers + 1):
        dec_list_for_trans1.append(globals()['lan1_decoder_layer' + str(i)])

    dec_layers_for_trans1 = nn.ModuleList(
        dec_list_for_trans1
    )

    customized_decoder_for_trans1 = ProposedDecoder(dec_layers_for_trans1)

    # encoder_layers for transformer2
    enc_list_for_trans2 = []
    for i in range(1, n_layers + 1):
        enc_list_for_trans2.append(globals()['lan2_encoder_layer' + str(i)])

    enc_layers_for_trans2 = nn.ModuleList(
        enc_list_for_trans2
    )

    customized_encoder_for_trans2 = ProposedEncoder(enc_layers_for_trans2)

    # decoder_layers for transformer2
    dec_list_for_trans2 = []
    for i in range(1, n_layers + 1):
        dec_list_for_trans2.append(globals()['lan2_decoder_layer' + str(i)])

    dec_layers_for_trans2 = nn.ModuleList(
        dec_list_for_trans2
    )

    customized_decoder_for_trans2 = ProposedDecoder(dec_layers_for_trans2)

    customized_transformer1 = torch.nn.Transformer(custom_encoder=customized_encoder_for_trans1,
                                                custom_decoder=customized_decoder_for_trans1,
                                                batch_first=batch_first)

    customized_transformer2 = torch.nn.Transformer(custom_encoder=customized_encoder_for_trans2,
                                                custom_decoder=customized_decoder_for_trans2,
                                                batch_first=batch_first)

    tok_embedding = nn.Embedding(vocab_size, d_model, 
                                      padding_idx=pad_idx)

    
    kwargs_transformer1 = {'tok_embedding_lan1': tok_embedding,
                            'tok_embedding_lan2': tok_embedding,
                            'transformer': customized_transformer1}
    wrap_transformer1 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, 
                                               max_len = max_length, device= device,**kwargs_transformer1)

    kwargs_transformer2 = {'tok_embedding_lan1': tok_embedding,
                            'tok_embedding_lan2': tok_embedding,
                            'transformer': customized_transformer2}
    wrap_transformer2 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, 
                                               max_len = max_length, device = device, **kwargs_transformer1)

    model = ProposedModel(wrap_transformer1, wrap_transformer2, d_model, vocab_size)

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
    # Part4. Desing Training Loop
    #####
    optimizer = AdamW(model.parameters(), lr = 5e-5)

    # lr scheduler
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    with torch.autograd.detect_anomaly():
        print("before strat training epoch, confirm")

    # model to device
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    writer = SummaryWriter()

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

            start_lan1_inter_prob, start_lan1_output_prob, start_lan2_inter_prob, start_lan2_output_prob\
                    = model(lan1_, lan2_, pad_idx, eos_idx)

            # start from lan1
            ## decoder prediction
            start_lan1_inter_prob_2d = start_lan1_inter_prob.contiguous().view(-1, start_lan1_inter_prob.shape[-1])
            ## decoder output(label) ->      k1 K2 K3 ... KN [EOS]
            tgt_start_lan1_inter = lan2_[:, 1:].contiguous().view(-1)
            loss_lan1_sample_lan2_vs_tgt_lan2 = cross_entropy(start_lan1_inter_prob_2d, tgt_start_lan1_inter)

            ## decoder prediction
            start_lan1_output_prob_2d = start_lan1_output_prob.contiguous().view(-1, start_lan1_output_prob.shape[-1])            
            ## decoder output(label)
            tgt_start_lan1_output = lan1_[:, 1:].contiguous().view(-1)
            loss_lan1_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan1_output_prob_2d, tgt_start_lan1_output)
            
            loss_lan1 = loss_lan1_sample_lan2_vs_tgt_lan2 + loss_lan1_sample_lan1_vs_tgt_lan1

            # start from lan2
            ## decoder prediction
            start_lan2_inter_prob_2d = start_lan2_inter_prob.contiguous().view(-1, start_lan2_inter_prob.shape[-1])
            # decoder output(label) ->      k1 k2 k3 ... kn [EOS]
            tgt_start_lan2_inter = lan1_[:, 1:].contiguous().view(-1)
            loss_lan2_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan2_inter_prob_2d, tgt_start_lan2_inter)
            
            start_lan2_output_prob_2d = start_lan2_output_prob.contiguous().view(-1, start_lan2_output_prob.shape[-1])
            tgt_start_lan2_output = lan2_[:, 1:].contiguous().view(-1)
            loss_lan2_smaple_lan2_vs_tgt_lan2 = cross_entropy(start_lan2_output_prob_2d, tgt_start_lan2_output)
            
            loss_lan2 = loss_lan2_sample_lan1_vs_tgt_lan1 + loss_lan2_smaple_lan2_vs_tgt_lan2
            
            loss_ = loss_lan1 + loss_lan2

            loss_.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            loss.append(loss_.item())
            loss_period.append(loss_.item())
            
            del batch

            if cnt % 100 == 0:
                # valid loss
                valid_loss = []

                model.eval()

                for v_b in eval_dataloader:
                    v_batch = {k: v.to(device) for k, v in v_b.items()}

                    lan1_ = v_batch['en']
                    lan2_ = v_batch['de']

                    start_lan1_inter_prob, start_lan1_output_prob, start_lan2_inter_prob, start_lan2_output_prob\
                            = model(lan1_, lan2_, pad_idx, eos_idx)

                    # start from lan1
                    ## decoder prediction
                    start_lan1_inter_prob_2d = start_lan1_inter_prob.contiguous().view(-1, start_lan1_inter_prob.shape[-1])
                    ## decoder output ->      k1 K2 K3 ... KN [EOS]
                    tgt_start_lan1_inter = lan2_[:, 1:].contiguous().view(-1)
                    loss_lan1_sample_lan2_vs_tgt_lan2 = cross_entropy(start_lan1_inter_prob_2d, tgt_start_lan1_inter)

                    ## decoder perdiction
                    start_lan1_output_prob_2d = start_lan1_output_prob.contiguous().view(-1, start_lan1_output_prob.shape[-1])            
                    ## decoder output
                    tgt_start_lan1_output = lan1_[:, 1:].contiguous().view(-1)
                    loss_lan1_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan1_output_prob_2d, tgt_start_lan1_output)
                    
                    loss_lan1 = loss_lan1_sample_lan2_vs_tgt_lan2 + loss_lan1_sample_lan1_vs_tgt_lan1

                    # start from lan2
                    ## decoder prediction
                    start_lan2_inter_prob_2d = start_lan2_inter_prob.contiguous().view(-1, start_lan2_inter_prob.shape[-1])
                    ## decoder output ->       k1 k2 k3  ... kn [EOS]
                    tgt_start_lan2_inter = lan1_[:, 1:].contiguous().view(-1)
                    loss_lan2_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan2_inter_prob_2d, tgt_start_lan2_inter)
                    
                    ## decoder prediction
                    start_lan2_output_prob_2d = start_lan2_output_prob.contiguous().view(-1, start_lan2_output_prob.shape[-1])
                    ## decoder output
                    tgt_start_lan2_output = lan2_[:, 1:].contiguous().view(-1)
                    loss_lan2_smaple_lan2_vs_tgt_lan2 = cross_entropy(start_lan2_output_prob_2d, tgt_start_lan2_output)
                    
                    loss_lan2 = loss_lan2_sample_lan1_vs_tgt_lan1 + loss_lan2_smaple_lan2_vs_tgt_lan2
                    
                    loss_ = loss_lan1 + loss_lan2

                    valid_loss.append(loss_.item())

                    del v_batch

                # Loss Reporting
                print("epoch: {}, cnt : {}, train loss: {}, valid loss: {}".format(epoch, cnt, np.mean(loss_period), np.mean(valid_loss)))
                train_loss_history.append(np.mean(loss_period))
                valid_loss_history.append(np.mean(valid_loss))
                
                if np.mean(valid_loss) < min_loss:
                    min_loss = np.mean(valid_loss)
                    torch.save(model, "./model/model_{}_{}_{}_{}.pt".format(epoch, cnt, np.mean(loss_period), np.mean(valid_loss)))

                # get back to training mode
                model.train()

                # free loss_period
                loss_period = []
            
            cnt += 1

        # Evaluate sacreBleu
        # model.eval()

    writer.close()
    
    # save
    torch.save(model, "./model/model_end.pt")
    
    with open("train_loss_history.pkl", "wb") as f:
        pickle.dump(train_loss_history, f)
    
    with open("valid_loss_history.pkl", "wb") as f:
        pickle.dump(valid_loss_history, f)