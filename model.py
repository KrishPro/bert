"""
Written by KrishPro @ KP

filename: `model.py`
"""

import torch.nn as nn
import torch
import math

class TokenEmbedding(nn.Module):
    def __init__(self, emb_size: int, vocab_size: int, maxlen: int = 5000, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.emb_size = emb_size
        
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        self.pos_embedding = pos_embedding

    def forward(self, tokens: torch.Tensor):
        token_embeddings: torch.Tensor = self.embedding(tokens.long()) * math.sqrt(self.emb_size)
        token_embeddings: torch.Tensor = token_embeddings + self.pos_embedding[:token_embeddings.size(0), :]
        return self.dropout(token_embeddings)


class Bert(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, vocab_size: int,
    dropout=0.1, pad_idx=0, activation='relu', layer_norm_eps=1e-5):
        super().__init__()
        
        self.embedding = TokenEmbedding(d_model, vocab_size, dropout=dropout, pad_idx=pad_idx)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps)
        encoder_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str):
        ckpt = torch.load(checkpoint_path)

        hparams = ckpt['hparams']
        state_dict = ckpt['state_dict']

        model = cls(**hparams)
        model.load_state_dict(state_dict)

        return model

    def forward(self, tokens: torch.Tensor):
        # tokens.shape: (S, N)
        token_embeddings: torch.Tensor = self.embedding(tokens)

        # token_embeddings.shape: (S, N, E)
        encoded_embeddings = self.encoder(token_embeddings)

        # encoded_embeddings.shape: (S, N, E)
        return encoded_embeddings

class BertLM(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int, vocab_size: int,
    dropout=0.1, pad_idx=0, activation='relu', layer_norm_eps=1e-5):
        super().__init__()
        
        self.bert = Bert(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, num_layers=num_layers, vocab_size=vocab_size,
        dropout=dropout, pad_idx=pad_idx, activation=activation, layer_norm_eps=layer_norm_eps)

        self.classifier = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor):
        # tokens.shape: (S, N)
        encoded_embeddings = self.bert(tokens)

        # encoded_embeddings.shape: (S, N, E)
        pred_tokens: torch.Tensor = self.classifier(encoded_embeddings)

        # pred_tokens.shape: (S, N, V)
        return pred_tokens


def test():
    bert = Bert(d_model=64, nhead=4, dim_feedforward=256, num_layers=3, vocab_size=30_000)

    bert_lm = BertLM(d_model=64, nhead=4, dim_feedforward=256, num_layers=3, vocab_size=30_000)

    src = torch.randint(low=0, high=30_000, size=(256, 8))

    print(f"input: {src.shape}")
    print()

    print(f"Bert output: {bert(src).shape}")
    print(f"BertLM output: {bert_lm(src).shape}")


if __name__ == '__main__':
    test()