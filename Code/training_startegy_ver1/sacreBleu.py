from model import *

import evaluate
import copy
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
import os

from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import get_scheduler
from transformers import DataCollatorForSeq2Seq
from datasets import load_dataset
from transformers import AutoTokenizer

from tokenizers import Tokenizer
from tokenizers.models import BPE

def preprocess(examples):
    # tok_inputs = tokenizer.encode(inputs).ids
    # tok_targets = tokenizer.encode(targets).ids
    tok_inputs = []
    tok_targets = []

    sos_id = tokenizer.token_to_id('[SOS]')
    eos_id = tokenizer.token_to_id('[EOS]')

    for example in examples['translation']:
        tok_input = tokenizer.encode(example['en']).ids
        tok_input.insert(0, sos_id) # ADD SOS TOKEN
        tok_input.append(eos_id) # ADD EOS TOKEN
        

        tok_target = tokenizer.encode(example['de']).ids
        tok_target.insert(0, sos_id) # ADD SOS TOKEN
        tok_target.append(eos_id) # ADD EOS TOKEN

    model_inputs = {'en':tok_inputs, 'de':tok_targets}

    return model_inputs

def reconstruct_sentence(input_en, input_de, max_token_len = 500):
    # decoder prediction - [SOS] t1 t2 t3 ...
    output_en = torch.tensor(sos_idx).view(1, 1)
    output_de = torch.tensor(sos_idx).view(1, 1)
    
    output_en = output_en.to(device)
    output_de = output_de.to(device)

    print(output_de)
    print(output_de.shape)
    # construct en -> de prediction
    ## mode: 'start1_from_lan1_to_lan2'
    for i in range(max_token_len):
        # predictions - (1, seq_len, vocab_size)
        if input_en.get_device() == output_de.get_device():
            print("oo")
        predictions = model(input_en, output_de, pad_idx, eos_idx, 'start1_from_lan1_to_lan2')
        # just extract the last token id
        prediction_id = torch.argmax(predictions, 2)[1, -1]

        # if prediction_id == [EOS], prediction end
        if prediction_id.item() == eos_idx:
            break
        
        prediction_id_2d = prediction_id.view(1, 1).to(device)
        output_de = torch.cat([output_de, prediction_id_2d], dim = 1)

    # construct de -> en prediction
    ## mode: 'start2_from_lan2_to_lan1'
    for i in range(max_token_len):
        # predictions - (1, seq_len, vocab_size)
        predictions = model(output_en, input_de, pad_idx, eos_idx, 'start2_from_lan2_to_lan1')
        # just extract the last token id
        prediction_id = torch.argmax(predictions, 2)[1, -1]

        # if prediction_id == [EOS], prediction end
        if prediction_id.item() == eos_idx:
            break
        
        prediction_id_2d = prediction_id.view(1, 1)
        output_en = torch.cat([output_en, prediction_id_2d], dim = 1)

    return output_en, output_de


if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    metric = evaluate.load('sacrebleu')
    
    # load  datasets
    raw_datasets = load_dataset('wmt16', 'de-en')

    # load tokenizer
    file_dict = {
        'tokenizer_30000': './WMT_16_TOKENIZER/wmt_16_30000.json',
        'tokenizer_20000': './WMT_16_TOKENIZER/wmt_16_20000.json',
        'tokenizer_10000': './WMT_16_TOKENIZER/wmt_16_10000.json',
    }

    with open(file_dict['tokenizer_10000'], 'r') as f:
        tokenizer = Tokenizer.from_str(''.join(f.readlines()))

    # define tokenized dataset
    valid_dataset = raw_datasets['validation']

    tokenized_ds = valid_dataset.map(
        preprocess,
        batched = True,
        remove_columns = valid_dataset.column_names,
    )

    # Define & Load Model
    model_file = '' #state_dict() file path
    model_data = torch.load(model_file)

    # define model
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
    max_length = 5000

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

    
    kwargs_transformer1 = {'tok_embedding': tok_embedding,
                            'transformer': customized_transformer1}
    wrap_transformer1 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, 
                                               max_len = max_length, device= device,**kwargs_transformer1)

    kwargs_transformer2 = {'tok_embedding': tok_embedding,
                            'transformer': customized_transformer2}
    wrap_transformer2 =  WrapperForTransformer(d_model = d_model, dropout_rate = 0.1, 
                                               max_len = max_length, device = device, **kwargs_transformer1)

    model = ProposedModel(wrap_transformer1, wrap_transformer2, d_model, vocab_size)

    # load model
    model.load_state_dict(model_data)

    # define dataloader
    eval_dataloader = DataLoader(
        tokenized_ds,
        batch_size = 1,
    )

    # define parameters
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')
    pad_idx = tokenizer.token_to_id('[PAD]')

    for b in eval_dataloader:
        batch = {k: torch.tensor(v).view(1, len(v)).to(device) for k, v in b.items()}
        
        en = batch['en'] #(1, english_seq_len)
        de = batch['de'] #(1, english_seq_len)
        
        en_decoder_output = en[:, 1:].contiguous().view(-1) #(1, english_seq_len - 1)
        de_decoder_output = de[:, 1:].contiguous().view(-1) #(1, german_seq_len - 1

        en_decoder_pred, de_decoder_pred = reconstruct_sentence(en, de, max_token_len = 500)

        del(batch)

        break

    