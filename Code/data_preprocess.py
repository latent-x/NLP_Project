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
        
        # except if sequence length exceed 2000
        if (len(tok_input) > 1000) | (len(tok_target) > 1000):
            pass
        else:
            tok_inputs.append(tok_input)
            tok_targets.append(tok_target)

    model_inputs = {'en':tok_inputs, 'de':tok_targets}

    return model_inputs

if __name__ == "__main__":
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

    # preprocess datasets
    tokenized_ds = raw_datasets.map(
        preprocess,
        batched = True,
        remove_columns = raw_datasets["train"].column_names,
    )

    # save preprocessed dataset
    tokenized_ds.save_to_disk('./WMT_16_PREPROCESSED')
