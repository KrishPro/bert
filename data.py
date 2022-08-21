"""
Written by KrishPro @ KP

filename: `data.py`
"""

import torch.utils.data as data
from tokenizers import Tokenizer
from utils import extract_num_sentences

class Dataset(data.Dataset):
    def __init__(self, data_path: str, vocab_path: str, chunk_size = 2**23) -> None:
        super().__init__()

        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_path)
        self.length = extract_num_sentences(data_path)
        self.data_file = open(data_path)
        self.chunk_size = chunk_size
        
        self._reset()

    def _load_chunk(self):

        chunk = (self.previous_left + self.data_file.read(self.chunk_size)).splitlines()

        self.previous_left = chunk[-1]

        chunk = chunk[:-1]

        if chunk == []:
            if self.previous_left == '':
                raise StopIteration
            else:
                chunk = [self.previous_left]
                self.previous_left = ''

        return chunk

    def _reset(self):
        self.current_chunk_start = 0
        self.data_file.seek(0)
        self.previous_left = ''

        self.current_chunk = self._load_chunk()

        self.current_chunk_end = len(self.current_chunk) - 1

    def load_chunk(self):
        self.current_chunk_start += len(self.current_chunk)
        self.current_chunk = self._load_chunk()
        self.current_chunk_end += len(self.current_chunk)


    def __getitem__(self, idx: int):
        if idx > self.current_chunk_end:
            self.load_chunk()

        sentence = self.current_chunk[idx - self.current_chunk_start]

        return self.tokenizer.encode(sentence).ids

    def __len__(self) -> int:
        return self.length