# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np

'''Recurrent Neural Network for Text Classification with Multi-Task Learning'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            # self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(config.embedding_pretrained).long(), freeze=False)
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed)
            self.embedding.weight = nn.Parameter(torch.from_numpy(self.config.embedding_pretrained), requires_grad=False)
        else:
            self.embedding = nn.Embedding(self.config.n_vocab, self.config.embed, padding_idx=self.config.n_vocab - 1)
        self.lstm = nn.LSTM(self.config.embed, self.config.hidden_size, self.config.num_layers,
                            bidirectional=True, batch_first=True, dropout=self.config.dropout)
        self.fc = nn.Linear(self.config.hidden_size * 2, self.config.num_classes)

    def forward(self, x):
        x = torch.tensor(x).long()
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
    # def forward(self, x):
    #     x, _ = x
    #     embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
    #     out, _ = self.lstm(embed)
    #     out = torch.cat((embed, out), 2)
    #     out = F.relu(out)
    #     out = out.permute(0, 2, 1)
    #     out = self.maxpool(out).squeeze()
    #     out = self.fc(out)
    #     return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out