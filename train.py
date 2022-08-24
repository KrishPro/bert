"""
Written by KrishPro @ KP

filename: `train.py`
"""

from typing import Tuple
from pytorch_lightning import LightningModule, Trainer
try:
    from model import BertLM
    from data import DataModule
except:
    from bert.model import BertLM
    from bert.data import DataModule
import torch.optim as optim
import torch.nn as nn
import torch

class Model(LightningModule):
    def __init__(self, d_model:int, nhead:int, dim_feedforward:int, num_layers:int, weight_decay=0.0, warmup_steps=4_000,
    base_lr=1, vocab_size=30_000, dropout=0.1, pad_idx=0, activation="gelu", layer_norm_eps=1e-5, print_logs=False) -> None:

        super().__init__()
        self.save_hyperparameters(ignore=["print_logs"])
        self.print_logs = print_logs

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

        loss: torch.Tensor = self.criterion(out.view(S*N, V), tgt.view(S*N))

        if self.print_logs: print(f"epoch={self.current_epoch} | batch_idx={batch_idx} | loss={loss.detach():.3f}")

        self.log("loss", loss.detach(), batch_size=N)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        src, tgt = batch
        # tgt.shape == src.shape == (S, N)
        S, N = tgt.shape

        out: torch.Tensor = self.bert_lm(src)
        # out.shape = (S, N, V)
        V = out.size(2)

        loss: torch.Tensor = self.criterion(out.reshape((S*N, V)), tgt.reshape((S*N)))

        self.log("val_loss", loss.detach(), batch_size=N, prog_bar=True)

        return loss

    def configure_optimizers(self):
        
        get_lr = lambda step: (self.hparams['d_model'] ** -0.5) * min((step+1) ** -0.5, (step+1)*(self.hparams['warmup_steps']**-1.5))

        optimizer = optim.Adam(self.parameters(), lr=self.hparams['base_lr'], weight_decay=self.hparams['weight_decay'], betas=(0.9, 0.98))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, get_lr)

        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]

def train(d_model:int, nhead:int, dim_feedforward:int, num_layers:int, data_dir:str, batch_size:int, weight_decay=0.0, warmup_steps=4_000, print_logs=False,
base_lr=1, vocab_size=30_000, dropout=0.1, pad_idx=0, activation="gelu", shuffle=False, pin_memory=False, use_workers=False, layer_norm_eps=1e-5, **kwargs):

    use_tpu = kwargs.get("accelerator") == "tpu"

    datamodule = DataModule(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle, pin_memory=pin_memory, use_workers=use_workers, use_tpu=use_tpu)

    model = Model(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers, weight_decay=weight_decay, print_logs=print_logs,
    vocab_size=vocab_size, dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps, warmup_steps=warmup_steps, base_lr=base_lr)

    trainer = Trainer(**kwargs)

    trainer.fit(model, datamodule)