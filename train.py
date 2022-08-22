"""
Written by KrishPro @ KP

filename: `train.py`
"""

from typing import Tuple
from pytorch_lightning import LightningModule, Trainer
from model import BertLM
from data import DataModule
import torch.optim as optim
import torch.nn as nn
import torch

class Model(LightningModule):
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int, num_layers:int, weight_decay=0.0, warmup_steps=4_000,
    vocab_size=30_000, dropout=0.1, pad_idx=0, activation="gelu", layer_norm_eps=1e-5) -> None:

        super().__init__()
        self.save_hyperparameters()


        self.bert_lm = BertLM(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers, vocab_size=vocab_size,
        dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps)

        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        src, tgt = batch
        # tgt.shape == src.shape == (S, N)
        S, N = tgt.shape

        out: torch.Tensor = self.bert_lm(src)
        # out.shape = (S, N, V)
        V = out.size(2)

        loss = self.criterion(out.view(S*N, V), tgt.view(S*N))

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        src, tgt = batch
        # tgt.shape == src.shape == (S, N)
        S, N = tgt.shape

        out: torch.Tensor = self.bert_lm(src)
        # out.shape = (S, N, V)
        V = out.size(2)

        loss = self.criterion(out.reshape((S*N, V)), tgt.reshape((S*N)))

        return loss

    def configure_optimizers(self):
        
        get_lr = lambda step: (self.hparams['d_model'] ** -0.5) * min((step+1) ** -0.5, (step+1)*(self.hparams['warmup_steps']**-1.5))

        optimizer = optim.Adam(self.parameters(), lr=1, weight_decay=self.hparams['weight_decay'])

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

def train(d_model:int, nhead:int, dim_feedforward:int, num_layers:int, data_dir:str, batch_size:int, weight_decay=0.0, warmup_steps=4_000,
vocab_size=30_000, dropout=0.1, pad_idx=0, activation="gelu", shuffle=False, pin_memory=False, use_workers=False, layer_norm_eps=1e-5, **kwargs):

    use_tpu = kwargs.get("accelerator") == "tpu"

    datamodule = DataModule(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, use_workers=use_workers, use_tpu=use_tpu)

    model = Model(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers, weight_decay=weight_decay,
    vocab_size=vocab_size, dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps, warmup_steps=warmup_steps)

    trainer = Trainer(**kwargs)

    trainer.fit(model, datamodule)