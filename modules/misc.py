import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from utils import sample_gaussian_tensors

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, d_in, d_h):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_h, bias=False)
        self.w_2 = nn.Linear(d_h, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        out = torch.bmm(attn.transpose(1, 2), x)
        if out.shape[1] == 1:
            out = out.squeeze(1)
        return out, attn

class UncertaintyModule(nn.Module):
    def __init__(self, d_in, d_out, d_h, agg):
        super().__init__()

        self.embed_dim = d_out
        self.agg = agg
        self.attention = MultiHeadSelfAttention(1, d_in, d_h)
        if self.agg == 'gru':
            self.rnn = nn.GRU(d_in, d_out//2, bidirectional=True, batch_first=True)
        else:
            self.fc1 = nn.Linear(d_in, d_out)
        self.fc = nn.Linear(d_in, d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x_cls, x, mask=None):
        x_attn, attn = self.attention(x, mask)
        if self.agg == 'gru':
            lens = torch.sum(~mask, dim=-1)
            packed = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
            if torch.cuda.device_count() > 1:
                self.rnn.flatten_parameters()
            x_rnn, _ = self.rnn(packed)
            padded = pad_packed_sequence(x_rnn, batch_first=True)
            last_hid_states = []
            for b in range(x.size(0)):
                last_hid_states.append(padded[0][b][lens[b] - 1, :])
            x_cls = torch.stack(last_hid_states)
        else:
            x_cls = self.fc1(x_cls)
        x = self.fc(x_attn) + x_cls
        return x, attn

class PIENet(nn.Module):
    def __init__(self, n_embed, d_in, d_out, d_h, dropout=0.0):
        super(PIENet, self).__init__()

        self.n_embed = n_embed
        self.attention = MultiHeadSelfAttention(n_embed, d_in, d_h)
        self.fc = nn.Linear(d_in, d_out)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x_cls, x, mask=None):
        residual, attn = self.attention(x, mask)
        residual = self.dropout(self.sigmoid(self.fc(residual)))
        if self.n_embed > 1:
            x_cls = x_cls.unsqueeze(1).repeat(1, self.n_embed, 1)
        x_cls = self.layer_norm(x_cls + residual)
        return x_cls, attn, residual

class PCMENet(nn.Module):
    def __init__(self, d_in, **kwargs):
        super(PCMENet, self).__init__()

        d_embed = kwargs['embed_dim']
        agg = 'gru' if d_in == kwargs['text_encoder_dim'] and kwargs['text_encoder'] == 'gru' else 'linear'

        self.n_embed = kwargs['n_embed']
        self.prob_embed = kwargs['prob_embed']
        if self.prob_embed:
            self.pie_net = PIENet(1, d_in, d_embed, d_in//2)
            self.uncertain_net = UncertaintyModule(d_in, d_embed, d_in//2, agg)
        else:
            self.pie_net = PIENet(self.n_embed, d_in, d_embed, d_in // 2)

    def forward(self, x_cls, x, mask=None):
        out = dict()
        x_attn, attn, res = self.pie_net(x_cls, x, mask)  # x_attn: LN(cls+residual) (b, n_emb, d_embed)
        out['attention'] = attn  # (b, n_emb)
        out['residual'] = res  # (b, n_emb, d_embed)  # residual is not normalized at this point
        x_attn = F.normalize(x_attn, p=2, dim=-1)
        if self.prob_embed:
            # TODO: Consider to FIX the variance
            logsig, attn = self.uncertain_net(torch.mean(x, dim=1), x, mask)  # logsig: no_LN(cls+residual) (b, 1, d_embed)
            out['logsigma'] = logsig
            out['sigma_attention'] = attn
            if self.n_embed > 1:
                out['mu_embeding'] = x_attn
                x = sample_gaussian_tensors(x_attn, logsig, self.n_embed)
            else:
                x = x_attn
        else:
            x = x_attn
        return x, out
