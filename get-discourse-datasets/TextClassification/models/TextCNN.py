import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if self.config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(self.config.embedding_pretrained, freeze=False)
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed)
            self.embedding.weight = nn.Parameter(torch.from_numpy(self.config.embedding_pretrained), requires_grad=False)
        else:
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed, padding_idx=self.config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.num_filters, (k, self.config.embed)) for k in self.config.filter_sizes])
        self.dropout = nn.Dropout(self.config.dropout)
        self.fc = nn.Linear(self.config.num_filters * len(self.config.filter_sizes), self.config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = torch.tensor(x).long()
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out