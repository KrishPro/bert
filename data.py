"""
Written by KrishPro @ KP

filename: `data.py`
"""

import random
from typing import List, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import torch.utils.data as data
from tokenizers import Encoding, Tokenizer
import os

class Dataset(data.Dataset):
    def __init__(self, data_path: str, vocab_path: str, chunk_size: int = 2 ** 10):
        # Extracting the length from the filename
        self.length = int(os.path.splitext(os.path.basename(data_path))[0].split('-')[-1])

        # Storing the data_path
        self.data_path = data_path

        # Loading the tokenizer
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_path)

        # Storing the chunk_size
        self.chunk_size = chunk_size

        self._reset()

    def _reset(self):
        # Re-opening the file
        if hasattr(self, 'file'): self.file.close()
        self.file = open(self.data_path)

        # Deleting current loaded chunk
        if hasattr(self, 'current_chunk'): del self.current_chunk

        # Reseting chunk_start_idx, previous_left
        self.chunk_start_idx = 0
        self.previous_left = ''

        # Loading the initial chunk
        self.load_chunk()


    def load_chunk(self):
        # Getting the size of current chunk
        current_chunk_size = len(self.current_chunk) if hasattr(self, 'current_chunk') and self.current_chunk else 0

        # Loading next chunk
        chunk = self.file.read(self.chunk_size)

        # If the current_chunk is blank, which means previous chunk was last. Then Re-set the dataset.
        if chunk == '':
            self._reset()
            return None

        # Adding the previous left-overs to the current_chunk
        self.current_chunk = self.previous_left + chunk

        # Spltting sentences
        self.current_chunk = self.current_chunk.split('\n')

        # Storing the current left-overs
        self.previous_left = self.current_chunk[-1]

        # Removing the current left-overs from the current chunk
        self.current_chunk = self.current_chunk[:-1]
        
        # Adding the size of previous chunk to the previous chunk_start_idx to get the current chunk_start_idx
        self.chunk_start_idx =  self.chunk_start_idx + current_chunk_size

    def mask(self, encoding: Encoding) -> Tuple[List[int], List[int]]:
        # Extracting ids
        ids: List[int] = encoding.ids

        # Creating blank lists
        src, tgt = list(), list()

        # Looping over ids
        for id in ids:
            # Sampling a random number b/w 0 and 1
            prob = random.random()

            # Masking 15% of tokens
            if prob < 0.15 and (id != self.tokenizer.token_to_id("[SOS]") and id != self.tokenizer.token_to_id("[EOS]")):
                prob /= 0.15

                # Replacing 80% tokens with [MASK] token
                if prob < 0.8:
                    src.append(self.tokenizer.token_to_id("[MASK]"))

                # Replacing 10% tokens with random token
                elif prob < 0.9:
                    src.append(random.randrange(self.tokenizer.get_vocab_size()))
                # Leaving 10% tokens un-changed
                else:
                    src.append(id)

                # Appending actual token to tgt
                tgt.append(id)
            
            else:
                # If the token is decided to be un-masked, the actual token and pad_id are appended to src and tgt respectively.
                src.append(id)
                tgt.append(self.tokenizer.padding['pad_id'])

        return src, tgt

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        # If the requested idx is not in the currently loaded chunk then, Raise IndexError
        if not (self.chunk_start_idx <= idx < (self.chunk_start_idx + len(self.current_chunk))):
            # Raise IndexError
            raise IndexError('list index out of range')
  
        # If the requested idx is the last item of the current_chunk, load the next chunk
        if (idx - self.chunk_start_idx) == (len(self.current_chunk) - 1):
            # Load the next chunk
            self.load_chunk()

        # Fetching the requested sentence from the current_chunk
        sentence: str = self.current_chunk[idx - self.chunk_start_idx]

        # Tokenizing the sentence
        encoding: Encoding = self.tokenizer.encode(sentence)

        # Masking random tokens
        src, tgt = self.mask(encoding)

        # Converting src & tgt to tensors before returning
        return torch.tensor(src), torch.tensor(tgt)

    @staticmethod
    def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
        src, tgt = zip(*data)

        # Padding src & tgt
        src = pad_sequence(list(src), padding_value=0)
        tgt = pad_sequence(list(tgt), padding_value=0)

        return src, tgt


    def __len__(self):
        # Returning the length, extracted from the filename
        return self.length
        