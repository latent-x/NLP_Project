from model import *

import copy
from tqdm.auto import tqdm

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
from datasets import load_dataset
from transformers import AutoTokenizer

from torch.utils.tensorboard import SummaryWriter

def preprocess_function_en_fr(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]

    model_inputs = tokenizer_en_fr(
        inputs, text_target = targets, max_length = max_length, truncation = True
    )
    return model_inputs

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #####
    # Part1. Preparing the data
    #####

    # parameters
    max_length = 5000

    raw_datasets = load_dataset('kde4', lang1 = "en", lang2 = "fr")

    split_datasets = raw_datasets["train"].train_test_split(train_size = 0.99, seed = 20)
    split_datasets["validation"] = split_datasets.pop("test")

    tokenizer_en_fr = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr", return_tensors = "pt")

    tokenized_datasets_en_fr = split_datasets.map(
        preprocess_function_en_fr,
        batched = True,
        remove_columns = split_datasets["train"].column_names,
    )

    #####
    # Part2. Define Model
    #####

    # parameters
    d_model = 512
    nhead = 8
    dropout_attention = 0.0
    batch_first = True
    n_layers = 6
    vocab_size = tokenizer_en_fr.vocab_size

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

    # define encoderlayer for language1
    for i in range(1, n_layers + 1):
        kwargs = {'self_attn': globals()['lan1_self_attention' + str(i)]}
        globals()['lan1_encoder_layer' + str(i)] = ProposedEncoderLayer(d_model = d_model, nhead = nhead, device = device, batch_first = batch_first, **kwargs)

    # define encoderlayer for language2
    for i in range(1, n_layers + 1):
        kwargs = {'self_attn': globals()['lan2_self_attention' + str(i)]}
        globals()['lan2_encoder_layer' + str(i)] = ProposedEncoderLayer(d_model = d_model, nhead = nhead, device = device, batch_first = batch_first, **kwargs)

    # define decoderlayer for language1
    for i in range(1, n_layers + 1):
        kwargs = {
            'self_attn': globals()['lan2_self_attention' + str(i)], 
            'multihead_attn': globals()['lan1_lan2_self_attention' + str(i)]
            }
        globals()['lan1_decoder_layer' + str(i)] = ProposedDecoderLayer(d_model = d_model, nhead = nhead, device = device, batch_first = batch_first, **kwargs)
        
    # define decoderlayer for langauge2
    for i in range(1, n_layers + 1):
        kwargs = {
            'self_attn': globals()['lan1_self_attention' + str(i)], 
            'multihead_attn': globals()['lan1_lan2_self_attention' + str(i)]
            }
        globals()['lan2_decoder_layer' + str(i)] = ProposedDecoderLayer(d_model = d_model, nhead = nhead, device = device, batch_first = batch_first, **kwargs)
        
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

    tok_embedding_lan1 = nn.Embedding(tokenizer_en_fr.vocab_size, d_model, padding_idx=tokenizer_en_fr.pad_token_id)
    tok_embedding_lan2 = nn.Embedding(tokenizer_en_fr.vocab_size, d_model, padding_idx=tokenizer_en_fr.pad_token_id)

    
    kwargs_transformer1 = {'tok_embedding_lan1': tok_embedding_lan1,
                            'tok_embedding_lan2': tok_embedding_lan2,
                            'transformer': customized_transformer1}
    wrap_transformer1 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, max_len = max_length, device= device,**kwargs_transformer1)

    kwargs_transformer2 = {'tok_embedding_lan1': tok_embedding_lan2,
                            'tok_embedding_lan2': tok_embedding_lan1,
                            'transformer': customized_transformer2}
    wrap_transformer2 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, max_len = max_length, device = device, **kwargs_transformer1)

    model = ProposedModel(wrap_transformer1, wrap_transformer2, d_model, vocab_size)

    #####
    # Part3. Define DataLoader
    #####
    data_collator_en_fr = DataCollatorForSeq2Seq(tokenizer_en_fr, model = model, padding = True)

    train_dataloader_en_fr = DataLoader(
        tokenized_datasets_en_fr["train"],
        shuffle = True,
        batch_size = 64,
        collate_fn = data_collator_en_fr,
    )

    eval_dataloader_en_fr = DataLoader(
        tokenized_datasets_en_fr["validation"],
        batch_size = 64,
        collate_fn = data_collator_en_fr,
    )

    #####
    # Part4. Desing Training Loop
    #####
    optimizer = AdamW(model.parameters(), lr = 5e-5)

    # lr scheduler
    num_epochs = 10
    num_training_steps = num_epochs * len(train_dataloader_en_fr)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer = optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # model to device
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))
    
    model.train()

    writer = SummaryWriter()

    cross_entropy = nn.CrossEntropyLoss(ignore_index = tokenizer_en_fr.pad_token_id)

    for epoch in range(num_epochs):
        loss = []
        for b in train_dataloader_en_fr:
            batch = {k: v.to(device) for k, v in b.items()}
            start_lan1_inter_prob, start_lan1_output_prob, start_lan2_inter_prob, start_lan2_output_prob\
                = model(batch['input_ids'], batch['labels'], tokenizer_en_fr.pad_token_id)
            # input_ids(src) == lan1
            # labels == lan2
            
            # start from lan1
            start_lan1_inter_prob_2d = start_lan1_inter_prob.contiguous().view(-1, start_lan1_inter_prob.shape[-1])
            tgt_start_lan1_inter = batch['labels'].contiguous().view(-1)
            loss_lan1_sample_lan2_vs_tgt_lan2 = cross_entropy(start_lan1_inter_prob_2d, tgt_start_lan1_inter)

            start_lan1_output_prob_2d = start_lan1_output_prob.contiguous().view(-1, start_lan1_output_prob.shape[-1])            
            tgt_start_lan1_output = batch['input_ids'].contiguous().view(-1)
            
            loss_lan1_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan1_output_prob_2d, tgt_start_lan1_output)
            loss_lan1 = loss_lan1_sample_lan2_vs_tgt_lan2 + loss_lan1_sample_lan1_vs_tgt_lan1

            # start from lan2
            start_lan2_inter_prob_2d = start_lan2_inter_prob.contiguous().view(-1, start_lan2_inter_prob.shape[-1])
            tgt_start_lan2_inter = batch['input_ids'].contiguous().view(-1)
            loss_lan2_sample_lan1_vs_tgt_lan1 = cross_entropy(start_lan2_inter_prob_2d, tgt_start_lan2_inter)
            
            start_lan2_output_prob_2d = start_lan2_output_prob.contiguous().view(-1, start_lan2_output_prob.shape[-1])
            tgt_start_lan2_output = batch['labels'].contiguous().view(-1)
            loss_lan2_smaple_lan2_vs_tgt_lan2 = cross_entropy(start_lan2_output_prob_2d, tgt_start_lan2_output)
            loss_lan2 = loss_lan2_sample_lan1_vs_tgt_lan1 + loss_lan2_smaple_lan2_vs_tgt_lan2
            
            loss = loss_lan1 + loss_lan2

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            loss.append(loss.item)

            del batch

        print("epoch: {}, loss: {}".format(epoch, np.mean(loss)))

    writer.close()
    torch.save(model, "test.pt")

