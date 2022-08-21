"""
Written by KrishPro @ KP

filename: `vocab.py`
"""

import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from utils import ensure_parent_dir

## IF YOU CHANGE THIS YOU MUST RE-CREATE ALL THE VOCAB FILES
## AS THIS VARIABLE AFFECTS ALL THOSE FILES
# 99.9% of sentences are shorter than 128
MAX_LEN = 256


def create_vocab(data_path: str, vocab_path: str = "./vocab.json"):
    """
    Reads the text file and creates a vocab out of it
    """
    # Creating the (bpe-based) base tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Setting up a pre_tokenizer
    tokenizer.pre_tokenizer = Whitespace()

    # Training the tokenizer on the vocab
    trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SOS]", "[EOS]", "[MASK]"])
    tokenizer.train(files=[data_path], trainer=trainer)

    # Adding the post-processing (Adding SOS & EOS tokens)
    tokenizer.post_processor = TemplateProcessing(single="[SOS] $A [EOS]",
        special_tokens=[
            ("[SOS]", tokenizer.token_to_id("[SOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]"))
        ]
    )

    # Enabling truncation
    tokenizer.enable_truncation(max_length=MAX_LEN)

    # Enabling padding
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")

    # Saving the output
    ensure_parent_dir(vocab_path)
    tokenizer.save(vocab_path)


def main(data_path: str, vocab_path: str):
    create_vocab(data_path, vocab_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data-path', '-i', type=str, required=True)
    parser.add_argument('--vocab-path', '-o', type=str, required=True)

    args = parser.parse_args()
    main(**args.__dict__)