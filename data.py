"""
Written by KrishPro @ KP

filename: `data.py`
"""

from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional

try: from prepare_data import PrepareData
except ImportError: from pytorch_bert.prepare_data import PrepareData

try: import torch_xla.core.xla_model as xm
except: pass

import torch.utils.data as data
import pytorch_lightning as pl
import torch
import os

class Dataset(data.Dataset):
    def __init__(self, data_dir: str):
        self.chunks = [os.path.join(data_dir, chunk) for chunk in os.listdir(data_dir)]

        self.length = sum(map(PrepareData._extract_num_sentences, self.chunks))

        self._reset()

    def _load_chunk(self, idx: int = None):
        if idx is None and hasattr(self, 'current_chunk_path'):
            idx = self.chunks.index(self.current_chunk_path) + 1

        if idx is None:
            print("Neither current_chunk_path nor idx was given, So setting idx to 0")
            idx = 0

        if -len(self.chunks) > idx or idx >= len(self.chunks):
            return None
   
        self.current_chunk_path = self.chunks[idx]
     
        with open(self.current_chunk_path) as file:
            chunk = file.read().split("\n")

        return chunk

    def _reset(self):
        self.current_chunk_start = 0

        self.current_chunk = self._load_chunk(idx=0)

        self.current_chunk_end = len(self.current_chunk) - 1

    def load_chunk(self):
        chunk = self._load_chunk()

        # If we've a chunk, Simply load it
        if chunk:
            self.current_chunk_start += len(self.current_chunk)
            self.current_chunk = chunk
            self.current_chunk_end += len(self.current_chunk)
        
        # If we don't, Reset the dataset and stop iterations
        else:
            self._reset()
            raise StopIteration

    @staticmethod
    def _parse_sentence(sentence: str):
        return list(map(int, sentence.split(' ')))

    def __getitem__(self, idx):
        if idx > self.current_chunk_end:
            self.load_chunk()

        assert self.current_chunk_start <= idx <= self.current_chunk_end, f"Index Error, Sampled out of chunk"

        item = self.current_chunk[idx - self.current_chunk_start]

        src, tgt = map(self._parse_sentence, item.split('\t'))

        return torch.tensor(src), torch.tensor(tgt)
    
    @staticmethod
    def collate_fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
        src, tgt = zip(*data)

        # Padding src & tgt
        src = pad_sequence(list(src), padding_value=0)
        tgt = pad_sequence(list(tgt), padding_value=0)

        return src, tgt
    
    def __len__(self):
        return self.length


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, use_workers: bool = False, pin_memory: bool = False, use_tpu: bool = False):
        super().__init__()
        # Storing the parameters we've got
        self.use_tpu = use_tpu
        self.data_dir = data_dir

        self._hparams = {'batch_size': batch_size, 'use_workers': use_workers, 'pin_memory': pin_memory, 'shuffle': False}

    def setup(self, stage: Optional[str] = None) -> None:
         # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            test_path, train_path = os.path.join(self.data_dir, 'test'), os.path.join(self.data_dir, 'train')

            self.train_dataset = Dataset(train_path)
            self.val_dataset = Dataset(test_path)

    def train_dataloader(self) -> None:
        # required for TPU support
        sampler = None
        if self.use_tpu:
            sampler = data.DistributedSampler(self.train_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=self._hparams['shuffle'])

        return data.DataLoader(self.train_dataset, batch_size=self._hparams['batch_size'], shuffle=self._hparams['shuffle'],
        pin_memory=self._hparams['pin_memory'], num_workers=os.cpu_count() if self._hparams['use_workers'] else 0, 
        collate_fn=Dataset.collate_fn, sampler=sampler)

    def val_dataloader(self) -> None:
        # required for TPU support
        sampler = None
        if self.use_tpu:
            sampler = data.DistributedSampler(
                self.val_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
            )
        return data.DataLoader(self.val_dataset, batch_size=self._hparams['batch_size'], shuffle=False,
        pin_memory=self._hparams['pin_memory'], num_workers=os.cpu_count() if self._hparams['use_workers'] else 0, 
        collate_fn=Dataset.collate_fn, sampler=sampler)