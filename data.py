"""
Written by KrishPro @ KP

filename: `data.py`
"""

import os
import torch
import random
import torch.utils.data as data
from tokenizers import Tokenizer, Encoding
from pytorch_lightning import LightningDataModule
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, List

try: from utils import extract_num_sentences
except: from bert.utils import extract_num_sentences

try: import torch_xla.core.xla_model as xm
except: pass

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


    def __getitem__(self, idx: int):
        if idx > self.current_chunk_end:
            self.load_chunk()

        sentence = self.current_chunk[idx - self.current_chunk_start]

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

    def __len__(self) -> int:
        return self.length

class DataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, shuffle=False, pin_memory=False, use_workers=False, use_tpu=False) -> None:
        super().__init__()

        self.dataloader_kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'pin_memory': pin_memory, 'num_workers': os.cpu_count() if use_workers else 0}
        self.sampler_kwargs = {'shuffle': shuffle}

        self.use_tpu = use_tpu
        self.data_dir = data_dir

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage == None:
            test, train = sorted([os.path.join(self.data_dir, p) for p in os.listdir(self.data_dir) if p.startswith("train") or p.startswith("test")])
            
            self.train_dataset = Dataset(train, os.path.join(self.data_dir, "vocab.json"))
            self.val_dataset = Dataset(test, os.path.join(self.data_dir, "vocab.json"))

        
    def train_dataloader(self):
        sampler = None
        
        if self.use_tpu:
            sampler = data.DistributedSampler(self.train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), **self.sampler_kwargs)

        return data.DataLoader(self.train_dataset, sampler=sampler, collate_fn=Dataset.collate_fn, **self.dataloader_kwargs)
    
    def val_dataloader(self):
        sampler = None
        
        if self.use_tpu:
            sampler = data.DistributedSampler(self.val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), **self.sampler_kwargs)

        return data.DataLoader(self.val_dataset, sampler=sampler, collate_fn=Dataset.collate_fn, **self.dataloader_kwargs)