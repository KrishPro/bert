"""
Written by KrishPro @ KP

filename: `data.py`
"""

from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Optional
from prepare_data import PrepareData
from tqdm import tqdm

try: import torch_xla.core.xla_model as xm
except: pass

import torch.utils.data as data
import pytorch_lightning as pl
import torch
import os

class Dataset(data.Dataset):
    def __init__(self, data_path: str, chunk_size: int = 2 ** 23):
        self.chunk_size = chunk_size
        self.file = open(data_path)

        try:
            self.length = PrepareData._extract_num_sentences(data_path)
        except:
            self.length = 0
            self.file.seek(0)
            for _ in self.file: self.length += 1
            self.file.seek(0)

        self._reset()

    def _load_chunk(self):
        if not hasattr(self, 'previous_left'): self.previous_left = ''
       
        chunk = (self.previous_left + self.file.read(self.chunk_size))
    
        last_new_line_char = chunk.rfind('\n')

        if last_new_line_char >= 0:
            self.previous_left = chunk[last_new_line_char+1:]
            chunk = chunk[:last_new_line_char]
            if not chunk:
                chunk = self.previous_left
                self.previous_left = ''
        else:
            self.previous_left = ''

        chunk = chunk.splitlines()

        try:
            chunk = list( map( lambda line: tuple(map(lambda s: list(map(int, s.strip().split(' '))), line.strip().split('\t'))), filter(lambda c: c and len(c.split('\t')) == 2, chunk) )) 
        except:
            try:
                final_chunk = []
                for c in chunk:
                    ss = c.strip().split('\t')
                    final_c = []
                    if len(ss) == 2:
                        for s in ss:
                            final_s = []
                            for i in s.strip().split(' '):
                                try:
                                    final_s.append(int(i))
                                except:
                                    print(i)
                            final_c.append(final_s)
                 
                        final_chunk.append(tuple(final_c))
                chunk = final_chunk
                del final_chunk, final_c, final_s, ss, c
            except:
                print("[WARN]: A chunk failed to load and is skipped. So, the dataset will have unexpectedly shorter length")
                chunk = self._load_chunk()

        return chunk

    def _reset(self):
        self.current_chunk_start = 0
        self.previous_left = ''
        self.file.seek(0)

        self.current_chunk = self._load_chunk()

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

    
    def __getitem__(self, idx):
        if idx == (self.current_chunk_end + 1):
            self.load_chunk()

        assert self.current_chunk_start <= idx <= self.current_chunk_end, f"Index Error, Sampled out of chunk"

        src, tgt = self.current_chunk[idx - self.current_chunk_start]

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
    def __init__(self, data_dir: str, batch_size: int, use_workers: bool, pin_memory: bool, use_tpu: bool = False, chunk_size: int = 2 ** 23):
        super().__init__()
        # Storing the parameters we've got
        self.use_tpu = use_tpu
        self.data_dir = data_dir

        self.dataset_kwargs = {'chunk_size': chunk_size}
        self._hparams = {'batch_size': batch_size, 'use_workers': use_workers, 'pin_memory': pin_memory, 'shuffle': False}

    def setup(self, stage: Optional[str] = None) -> None:
         # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:

            test_path, train_path = [os.path.join(self.data_dir, path) for path in sorted(os.listdir(self.data_dir)) if path.startswith('train') or path.startswith('test')]

            self.train_dataset = Dataset(train_path, **self.dataset_kwargs)
            self.val_dataset = Dataset(test_path, **self.dataset_kwargs)

    @staticmethod
    def train_test_split(input_path: str, output_dir: str, val_ratio: float = 0.01):
        """
        This function will perform a train-test split
        """
        num_sentences = PrepareData._extract_num_sentences(input_path)
        val_size = int(num_sentences * val_ratio)
        
        with open(input_path) as file:

            train_file_path, test_file_path = os.path.join(output_dir, f'train-{num_sentences-val_size}.txt'), os.path.join(output_dir, f'test-{val_size}.txt')

            with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file:
                for i, sentence in enumerate(tqdm(file, total=num_sentences)):
                    if i < val_size:
                        test_file.write(sentence)
                    else:
                        train_file.write(sentence)

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