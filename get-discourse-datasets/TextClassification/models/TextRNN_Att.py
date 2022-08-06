import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed)
            self.embedding.weight = nn.Parameter(torch.from_numpy(self.config.embedding_pretrained), requires_grad=False)
        else:
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed, padding_idx=self.config.n_vocab - 1)
        self.lstm = nn.LSTM(self.config.embed, self.config.hidden_size, self.config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(self.config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(self.config.hidden_size * 2, self.config.hidden_size2)
        self.fc = nn.Linear(self.config.hidden_size2, self.config.num_classes)

    def forward(self, x):
        x = torch.tensor(x).long()
        # x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out