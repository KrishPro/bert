"""
Written by KrishPro @ KP

filename: `train.py`
"""


import shutil

import os
import torch
import torch.nn as nn
from typing import Tuple
import torch.optim as optim
from tokenizers import Tokenizer
import pytorch_lightning as pl

try:
    from data import DataModule
    from model import BertLM
except:
    from pytorch_bert.data import DataModule
    from pytorch_bert.model import BertLM


class Model(pl.LightningModule):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, vocab_size: int,
    warmup_steps=4000, dropout=0.1, pad_idx=0, activation='relu', layer_norm_eps=1e-5):
        
        super().__init__()
        self.save_hyperparameters(ignore=['warmup_steps'])
        self.warmup_steps = warmup_steps

        self.bert_lm = BertLM(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,  num_layers=num_layers, vocab_size=vocab_size,
        dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        src, tgt = batch

        # src.shape: (S, N)
        # tgt.shape: (S, N)

        assert src.shape == tgt.shape
        S, N = src.shape

        out: torch.Tensor = self.bert_lm(src)
        # out.shape: (S, N, V)
        _, _, V = out.shape

        loss: torch.Tensor = self.criterion(out.reshape(S*N, V), tgt.reshape(S*N))

        self.log("loss", loss.detach(), batch_size=N)
        
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        src, tgt = batch

        # src.shape: (S, N)
        # tgt.shape: (S, N)

        assert src.shape == tgt.shape
        S, N = src.shape

        out: torch.Tensor = self.bert_lm(src)
        # out.shape: (S, N, V)
        _, _, V = out.shape

        loss: torch.Tensor = self.criterion(out.reshape(S*N, V), tgt.reshape(S*N))

        self.log("val_loss", loss.detach(), batch_size=N)
        
        return loss

    def configure_optimizers(self):
        
        get_lr = lambda step: (self.hparams['d_model'] ** -0.5) * min((step+1) ** -0.5, (step+1)*(self.warmup_steps**-1.5))

        optimizer = optim.Adam(self.parameters(), lr=1)

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]


def train(d_model: int, nhead: int, dim_feedforward: int, num_layers: int, epochs: int, batch_size: int, use_workers: bool, pin_memory: bool,
data_dir: str, dropout=0.1, activation='relu', layer_norm_eps=1e-5, chunk_size=2**23, warmup_steps=4000, **kwargs):

    tokenizer: Tokenizer = Tokenizer.from_file(os.path.join(data_dir, 'vocab.json'))
    vocab_size: int = tokenizer.get_vocab_size()
    pad_idx = tokenizer.token_to_id("[PAD]")

    datamodule = DataModule(data_dir, os.path.join(data_dir, 'vocab.json'), batch_size=batch_size,
    use_workers=use_workers, pin_memory=pin_memory, chunk_size=chunk_size)

    model = Model(d_model, nhead, dim_feedforward, num_layers, vocab_size, warmup_steps=warmup_steps, dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps)

    trainer = pl.Trainer(max_epochs=epochs, **kwargs)

    trainer.fit(model, datamodule)

    ckpt_dir = os.path.join('lightning_logs', sorted(map(lambda name: (int(name.split('_')[1]), name), os.listdir('lightning_logs')))[-1][1], 'checkpoints')
    if os.path.exists(ckpt_dir):
        ckpt_path = os.path.join(ckpt_dir, sorted(os.listdir(ckpt_dir))[-1])

        lightning_ckpt = torch.load(ckpt_path)

        hparams = lightning_ckpt['hyper_parameters']
        state_dict = lightning_ckpt['state_dict']

        state_dict = {k[13:]: v for k, v in state_dict.items() if k.startswith('bert_lm.bert')}

        torch.save({'hparams': hparams, 'state_dict': state_dict}, 'output.ckpt')

        shutil.rmtree(ckpt_dir)
        

if __name__ == '__main__':
    
    # The below example will raise error, as the data_dir doesn't exists
    train(d_model=64, nhead=8, dim_feedforward=256, num_layers=3, epochs=10, batch_size=32, use_workers=True, pin_memory=True,
    data_dir='.ignore', dropout=0.1, activation='relu', limit_train_batches=5, limit_val_batches=2)