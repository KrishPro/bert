"""
Written by KrishPro @ KP

filename: `data.py`
"""

import random
from typing import List, Optional, Tuple
import torch
import pytorch_lightning as pl
from torch.nn.utils.rnn import pad_sequence
import torch_xla.core.xla_model as xm
import torch.utils.data as data
from tokenizers import Encoding, Tokenizer
import os


from tqdm import tqdm

class Dataset(data.Dataset):
    def __init__(self, data_path: str, vocab_path: str, chunk_size: int = 2 ** 10):
        super().__init__()
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
            
            # If the requested idx is not from the currently loaded chunk, a random item is returned
            idx = random.randint(self.chunk_start_idx, self.chunk_start_idx+len(self.current_chunk)-1)
        
            # # Raise IndexError
            # raise IndexError('list index out of range')
  
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
        

class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, vocab_path: str, batch_size: int, use_workers: bool, pin_memory: bool, use_tpu: bool = False, chunk_size: int = 2 ** 23):
        super().__init__()
        # Storing the parameters we've got
        self.use_tpu = use_tpu
        self.data_dir = data_dir

        self.dataset_kwargs = {'vocab_path': vocab_path, 'chunk_size': chunk_size}
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
        num_sentences = int(os.path.splitext(os.path.basename(input_path))[0].split('-')[-1])
        val_size = int(num_sentences * val_ratio)
        file = open(input_path)

        train_file_path, test_file_path = os.path.join(output_dir, f'train-{num_sentences-val_size}.txt'), os.path.join(output_dir, f'test-{val_size}.txt')

        with open(train_file_path, 'w') as train_file, open(test_file_path, 'w') as test_file:
            for i, sentence in enumerate(tqdm(file, total=num_sentences)):
                if i < val_size:
                    test_file.write(sentence)
                else:
                    train_file.write(sentence)

        file.close()

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
