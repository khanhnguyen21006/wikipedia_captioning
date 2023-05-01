import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from utils import sample_gaussian_tensors

def get_image_pooler(dim, _config):
    """
    Pooling strategies experimented:
        - PCME/PVSE
        - Slot Attention
    """
    _name = _config['image_pooling']
    if _name in ['pcme', 'pvse']:
        model = PCMENet(dim, **_config)
    elif _name == 'slot':
        model = SlotAttention(dim, **_config)
    else:
        raise ValueError(f"{_name} Image Pooling is not supported.")
    return model

def get_text_pooler(dim, _config):
    """
    Pooling strategies experimented:
        - PCME/PVSE
        - Slot Attention
        - Gaussian Cross Attention
    """
    _name = _config['text_pooling']
    if _name in ['pcme', 'pvse']:
        model = PCMENet(dim, **_config)
    elif _name == 'slot':
        model = SlotAttentionNet(dim, **_config)
    elif _name == 'gaussian':
        model = GaussianAttentionNet(dim, **_config)
    else:
        raise ValueError(f"{_name} Text Pooling is not supported.")
    return model

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
            out['mu_embeding'] = x_attn
            out['logsigma'] = logsig
            out['sigma_attention'] = attn
            if self.n_embed > 1:
                x = sample_gaussian_tensors(x_attn, logsig, self.n_embed)
            else:
                x = x_attn
        else:
            x = x_attn
        return x, out

class SlotAttentionNet(nn.Module):
    """Slot Attention https://arxiv.org/abs/2006.15055"""
    def __init__(self, d_in, **kwargs):
        super().__init__()
        self.T = kwargs['n_slot_itertation']
        self.K = kwargs['n_slot']
        d_emb = kwargs['embed_dim']
        self.d_emb = d_emd
        self.scale = d_emb ** -0.5

        self.ln_cls = nn.LayerNorm(d_emb)
        self.ln_q = nn.LayerNorm(d_emb)
        self.ln_kv = nn.LayerNorm(d_in)
        self.ln_pre_mlp = nn.LayerNorm(d_emb)
        self.ln_post_mlp = nn.LayerNorm(d_emb)

        # Parameters for Gaussian init (shared by all slots).
        self.q_mu = nn.Parameter(torch.randn(1, 1, d_emb))
        self.q_logsig = nn.Parameter(torch.zeros(1, 1, d_emb))
        init.xavier_uniform_(self.q_logsig)

        # Linear maps for the attention module.
        self.w_q = nn.Linear(d_in, d_emb)
        self.w_k = nn.Linear(d_in, d_emb)
        self.w_v = nn.Linear(d_in, d_emb)

        # Slot update functions.
        self.w_u = nn.Linear(d_emb, d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb)
        )

    def forward(self, x_cls, x, eps=1e-8):
        x_in = self.ln_kv(x)  # Apply layer norm to the input.
        k = self.w_k(x_in)  # (bs, N, d_emb)
        v = self.w_v(x_in)  # (bs, N, d_emb)

        x = sample_gaussian_tensors(self.q_mu, self.q_logsig, self.K)  # (bs, K, d_emb)
        for _ in range(self.T): # Multiple rounds of attention.
            x_prev = x  # (bs, K, d_emb)
            x = self.ln_q(x) # (bs, K, d_emb)

            # Attention.
            q = self.w_q(x)  # (bs, K, d_emb)
            attn = F.softmax(torch.bmm(k, q.transpose(2, 1))/(self.scale), dim=-1)  # (bs, N, K)
            attn += eps
            attn /= torch.sum(attn, dim=1).unsqueeze(1)  # (bs, N, K)
            x = torch.bmm(attn.transpose(2, 1), v)  # (bs, K, d_emb)

            # Slot update.
            x = self.w_u(x) + x_prev
            x = self.mlp(self.ln_mlp(x)) + x

            x = self.ln_post_mlp(x + x_cls.unsqueeze(1).repeat(1, self.K, 1))
        return x

class GaussianAttention(nn.Module):
    def __init__(self, d_in, **kwargs):
        super().__init__()

    def forward(self, x_cls, x):
        pass
