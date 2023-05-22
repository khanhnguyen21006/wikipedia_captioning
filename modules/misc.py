import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import copy
from functools import reduce
import numpy as np
from utils import sample_gaussian_tensors, log_mixture_gaussian, gelu

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
        model = SlotAttentionNet(dim, **_config)
    else:
        raise ValueError(f"{_name} Image Pooling is not supported.")
    return model

def get_text_pooler(dim, _config):
    """
    Pooling strategies experimented:
        - PCME/PVSE
        - Slot Attention
    """
    _name = _config['text_pooling']
    if _name in ['pcme', 'pvse']:
        model = PCMENet(dim, **_config)
    elif _name == 'slot':
        model = SlotAttentionNet(dim, **_config)
    else:
        raise ValueError(f"{_name} Text Pooling is not supported.")
    return model

def get_fuser(_config):
    """
        - Multi-Modal Transformers
        - Gaussian Cross Attention
    """
    _name = _config['fuse']
    if _name == 'mmbert':
        model = MMBert(dim)
    elif _name == None:
        model = None
    elif _name == 'gaussian':
        model = GaussianBert(dim)
    else:
        raise ValueError(f"{_name} Fuser is not supported.")
    return model

def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n)])

class MultiHeadSelfAttention(nn.Module):
    """Self-Attention layer introduced in https://arxiv.org/abs/1703.03130"""
    def __init__(self, n_head, d_in, d_h):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_h, bias=False)
        self.w_2 = nn.Linear(d_h, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        attn = self.w_2(self.tanh(self.w_1(x)))  # (len, d_in)->(len, d_h)->(len, n_head)
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        out = torch.bmm(attn.transpose(1, 2), x) # (n_head, len)x(len, d_in)->(n_head, d_in)
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
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x_cls, x, pad_mask=None):
        x_attn, attn = self.attention(x, mask)
        if self.agg == 'gru':
            lens = torch.sum(~pad_mask, dim=-1)
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
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x_cls, x, pad_mask=None):
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
        pad_mask = ~mask if mask is not None else None
        x_attn, attn, res = self.pie_net(x_cls, x, pad_mask)  # x_attn: LN(cls+residual) (b, n_emb, d_embed)
        out['attention'] = attn  # (b, len, n_emb)
        out['residual'] = res  # (b, n_emb, d_embed)  # residual is not normalized at this point
        x_attn = F.normalize(x_attn, p=2, dim=-1)
        if self.prob_embed:
            # TODO: Consider to FIX the variance
            logsig, attn = self.uncertain_net(torch.mean(x, dim=1), x, pad_mask)  # logsig: no_LN(cls+residual) (b, 1, d_embed)
            out['mu_embeding'] = x_attn
            out['logsigma'] = logsig
            out['sigma_attention'] = attn
            if self.n_embed > 1:
                x = sample_gaussian_tensors(x_attn, logsig, self.n_embed)
            else:
                x = x_attn
        else:
            x = x_attn
        return x, x_mask, out

class ModifiedSlotAttention(nn.Module):
    """Modified Slot Attention from https://arxiv.org/abs/2006.15055"""
    def __init__(self, d_in, d_emb, **kwargs):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_emb)
        self.ln_kv = nn.LayerNorm(d_in)
        self.ln_mlp = nn.LayerNorm(d_emb)
        self.scale = d_emb ** -0.5
        # Linear maps for the attention module.
        self.w_q = nn.Linear(d_emb, d_emb)
        self.w_k = nn.Linear(d_in, d_emb)
        self.w_v = nn.Linear(d_in, d_emb)
        # Slot update functions.
        self.w_u = nn.Linear(d_emb, d_emb)
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb)
        )
        # self.apply(self._init_weights())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def _pos_embed(self, x, pos=None):
        return x if pos is None else x + pos

    def forward(self, q_prev, memory, q_pos=None, mem_pos=None, mem_mask=None, eps=1e-9):
        q = self._pos_embed(self.ln_q(q_prev), q_pos)
        q = self.w_q(q)  # (bs, K, d_emb)
        kv = self._pos_embed(self.ln_kv(memory), mem_pos)
        k = self.w_k(kv)  # (bs, N, d_emb)
        v = self.w_v(kv)  # (bs, N, d_emb)

        # Attention.
        attn = F.softmax(torch.bmm(k, q.transpose(1, 2)) * self.scale, dim=-1) + eps  # (bs, N, K)
        if mem_mask is not None:
            kv_mask = mem_mask.unsqueeze(2).repeat(1, 1, q.size(1))  # (bs, N, K)
            attn.masked_fill_(~kv_mask, 0.)
        attn = attn/torch.sum(attn, dim=1).unsqueeze(1)  # (bs, N, K)
        x = torch.bmm(attn.transpose(1, 2), v)  # (bs, K, d_emb)

        # Slot update.
        x = self.w_u(x) + q_prev
        x = self.mlp(self.ln_mlp(x)) + x
        return x

class SlotAttentionNet(nn.Module):
    def __init__(self, d_in, **kwargs):
        super().__init__()
        self.T = 4
        self.K = kwargs['n_embed']
        d_emb = kwargs['embed_dim']
        self.ln_cls = nn.LayerNorm(d_emb)
        self.q_embed = nn.Embedding(self.K, d_emb)
        self.kvpos_embed = PositionEmbedding(d_in, normalize=True)

        attn_layer = ModifiedSlotAttention(d_in, d_emb)
        self.layers = _get_clones(attn_layer, self.T)
        # self.apply(self._init_weights())

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x_cls, x, mask=None):
        out = dict()
        if mask is None:
            mask = torch.ones(x.shape[:2]).bool().to(x.device)
        kv_pos = self.kvpos_embed(x, mask)
        q_embed = self.q_embed.weight.unsqueeze(0).repeat(x.size(0), 1, 1)
        q_prev = torch.zeros_like(q_embed)
        for layer in self.layers:
            q_prev = layer(q_prev, x, q_pos=q_embed, mem_pos=kv_pos, mem_mask=mask)
        out['residual'] = q_prev
        x = q_prev + self.ln_cls(x_cls.unsqueeze(1).repeat(1, self.K, 1))
        return x, out

class PositionEmbedding(nn.Module):
    """
    This is a adapted version of the position embedding
    Based from https://github.com/facebookresearch/detr.
    """
    def __init__(self, d_emb, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.d_emb = d_emb
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * torch.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None and mask.size() == (x.size(0), x.size(1))  # (bs, len)
        pos = mask.cumsum(1, dtype=torch.float32) # (bs, len)
        if self.normalize:
            eps = 1e-6
            pos = pos / (pos[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.d_emb, dtype=torch.float32, device=x.device)  # (d_emb)
        dim_t = self.temperature ** (2 * dim_t / self.d_emb)

        pos_embed = pos[:, :, None] / dim_t  # (bs, len, d_emb)
        pos_embed[:, :, 0::2] = pos_embed[:, :, 0::2].sin()
        pos_embed[:, :, 1::2] = pos_embed[:, :, 1::2].cos()
        return pos_embed

def dot_product_attention(q, k, v, kv_mask=None, drop=None, attn_only=False):
    attn = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)  # (b, h, n, d_h) x (b, h, d_h, m) => (b, h, n, m)
    if kv_mask is not None:
        attn = attn.masked_fill(~kv_mask, -1e9)
    attn = F.softmax(attn, dim=-1)
    if drop is not None:
        attn = drop(attn)
    if attn_only:
        return attn
    x = torch.matmul(attn, v)  # (b, h, n, m) x (b, h, m, d_h) => (b, h, n, d_h)
    return x, attn

class SimplifiedBert(nn.Module):
    def __init__(self, d_emb, **kwargs):
        super().__init__()
        n_layer = 5
        n_head = 16
        dropout = 0.1

        self.mode_embed = nn.Embedding(2, d_emb)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.2)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _prepare_input(self, out):
        image_cls = out['image']['cls_embedding']
        text_cls = out['section']['cls_embedding']
        image_mode = self.mode_embed(torch.zeros_like(image_cls))
        text_mode = self.mode_embed(torch.ones_like(text_cls))

        out['image']['embedding'] = out['image']['embedding'] + image_mode.unsqueeze(1)
        out['image']['cls_embedding'] = image_cls + image_mode
        out['section']['embedding'] = out['section']['embedding'] + text_mode.unsqueeze(1)
        out['section']['cls_embedding'] = out['section']['cls_embedding'] + text_mode

    def _prepare_output(self, out):
        pass

class GaussianAttentionLayer(nn.Module):
    """Following https://arxiv.org/abs/2109.05244"""
    def __init__(self, d_emb, h, K, dropout):
        super().__init__()
        assert d_emb % h == 0
        self.d_h = d_emb // h
        self.h = h
        self.K = K
        self.ln_kqv = nn.ModuleList([nn.LayerNorm(d_emb, d_emb) for _ in range(3)])
        self.w_kqv = nn.ModuleList([nn.Linear(d_emb, d_emb) for _ in range(3)])
        self.drop_attn = nn.Dropout(p=dropout)

        self.w_gauss = nn.ModuleList([nn.Linear(d_h, d_h) for _ in range(3)])
        self.v_gauss = nn.ModuleList([nn.Linear(d_h, K) for _ in range(3)])
        self.w_gate = nn.Linear(d_h, d_h)
        self.v_gate = nn.Linear(d_h, 1)

        self.w_u = nn.Linear(d_emb, d_emb)
        self.drop_u = nn.Dropout(p=dropout)
        self.ln_u = nn.Linear(d_emb, d_emb)

    def gaussian_attention(self, q, kv_mask):
        om_hat, mu_hat, sig_hat = [
            lpv(F.tanh(lpw(t)))  # d_h => d_h => K
            for lpw, lpv, t in zip(self.w_gauss, self.v_gauss, [q, q, q])
        ]

        kv_len = torch.sum(kv_mask, dim=-1)  # (b)
        om = F.softmax(om_hat, dim=-1)  # (b, h, n, K)
        mu = F.sigmoid(mu_hat, dim=-1) * kv_len[:, None, None, None]  # (b, h, n, K)
        logsig = torch.log(reduce(torch.min, [
            kv_len[:, None, None, None]/6 * F.sigmoid(sig_hat),
            mu/3 , (kv_len[:, None, None, None] - mu)/3  
        ]))
        
        x = kv_mask.cumsum(1, dtype=torch.float32) # (bs, m)
        attn = log_1d_mixture_gaussian(x, om, mu, logsig) # (bs, h, n, m)
        attn = attn.masked_fill(~kv_mask[:, None, None, :], 0.)
        return attn

    def forward(self, x, cntx, cntx_mask=None):
        _bs, _, d = x.size()  # (b, n, d_emb)
        q, k, v = [
            lp(ln(t)).view(_bs, -1, self.h, self.d_h).transpose(1, 2)  # d_emb => h x d_h
            for lp, ln, t in zip(self.w_kqv, self.ln_kqv, [x, cntx, cntx])
        ]

        # Attention
        c_attn = dot_product_attention(q, k, v, kv_mask=cntx_mask, drop=self.drop_attn, attn_only=True)  # (b, h, n, m)
        g_attn = self.gaussian_attention(q, cntx_mask) # (b, h, n, m)
        gate = F.sigmoid(self.v_gate(F.tanh(self.w_gate(q))))  # (b, h, n, 1)
        attn = (1 - gate) * c_attn + gate * g_attn
        x_attn = torch.matmul(attn, v)

        # Concat => transpose => project(dropout) => layer norm
        x = self.w_u(x_attn.transpose(1, 2).contiguous().view(_bs, -1, self.h * self.d_h))
        if self.drop_u is not None:
            x = self.drop_u(x)
        x = self.ln_u(x + x_attn)
        return x

class GaussianBertLayer(nn.Module):
    def __init__(self, d_emb, h, K, dropout):
        super().__init__()
        self.im_gauss_attn = GaussianAttentionLayer(d_emb, n_head, K, dropout)
        self.txt_gauss_attn = GaussianAttentionLayer(d_emb, n_head, K, n_dropout)
        # self.gauss_attn = GaussianAttentionLayer(d_emb, n_head, dropout)

        self.im_self_attn = BertAttentionLayer(d_emb, n_head, dropout)
        self.txt_self_attn = BertAttentionLayer(d_emb, n_head, dropout)

        # Intermediate and Output Layers (FFNs)
        self.im_inter = BertIntermediate(d_emb)
        self.im_output = BertOutput(d_emb)
        self.txt_inter = BertIntermediate(d_emb)
        self.txt_output = BertOutput(d_emb)

    def forward(self, x, mask=None):
        assert isinstance(x, tuple)  
        assert (mask is not None and isinstance(x_mask, tuple) and len(x) == len(mask))

        im_cls, im_emb, txt_cls, txt_emb = x
        im_cls_mask, im_emb_mask, txt_cls_mask, txt_emb_mask = x_mask
        im, im_mask = torch.cat([im_cls, im_emb], dim=1), torch.cat([im_cls_mask, im_emb_mask], dim=1)
        txt, txt_mask = torch.cat([txt_cls, txt_emb], dim=1), torch.cat([txt_cls_mask, txt_emb_mask], dim=1)
        _bs = im_emb.shape[0]

        im_xattn = self.im_gauss_attn(im, txt, cntx_mask=txt_mask)
        txt_xattn = self.txt_gauss_attn(txt, im, cntx_mask=im_mask)

        im_sattn = self.im_self_attn(im_xattn, im_xattn, cntx_mask=im_mask)
        txt_sattn = self.txt_self_attn(txt_xattn, txt_xattn, cntx_mask=txt_mask)
        
        im_inter_output = self.im_inter(im_sattn)
        txt_inter_output = self.txt_inter(txt_sattn)

        # Layer output
        im_output = self.im_output(im_inter_output, im_emb)
        txt_output = self.txt_output(txt_inter_output, txt_emb)

        return im_output[:, 0, :], im_output[:, 1:, :], txt_output[:, 0, :], txt_output[:, 1:, :]

class GaussianBert(SimplifiedBert):
    def __init__(self, d_emb, **kwargs):
        super().__init__()
        K = 5
        self.attn_layers = _get_clones(
            GaussianBertLayer(d_emb, n_head, K, dropout), n_layer
        )
        # self.apply(self._init_weights())

    def prepare_output(self, x, out):
        out['image']['embedding'], out['image']['cls_embedding'], \
                out['section']['embedding'], out['section']['cls_embedding'] = x

    def prepare_input(self, out):
        self._prepare_input(out)
        x = (
            out['image']['cls_embedding'], out['image']['embedding'],
            out['section']['cls_embedding'], out['section']['embedding']
        )
        x_mask = (
            torch.ones((_bs, 1), device=x[0].device).bool(),
            out['image']['mask'].bool(),
            torch.ones((_bs, 1), device=x[2].device).bool(),
            out['section']['mask'].bool()
        )
        return x, x_mask

    def forward(self, out, return_states=False):
        x, x_mask = self.prepare_input(out)
        all_states = []
        x_prev = x
        for layer in self.layers:
            x_prev = layer(x_prev, mask=x_mask)
            if return_states:
                all_states.append(x_prev)
        self.prepare_output(x_prev, out)
        if return_states:
            return out, all_states
        return out

class BertAttentionLayer(nn.Module):
    """ 
        A single attention layer following BERT with simplified configs,
        equivalent to (BertAttention + BertAttOutput) in pytorch transfomers implementation.
    """
    def __init__(self, d_emb, h, dropout=0.1):
        super().__init__()
        assert d_emb % h == 0
        self.d_h = d_emb // h
        self.h = h
        self.ln_kqv = nn.ModuleList([nn.LayerNorm(d_emb, d_emb) for _ in range(3)])
        self.w_kqv = nn.ModuleList([nn.Linear(d_emb, d_emb) for _ in range(3)])
        self.drop_attn = nn.Dropout(p=dropout)
        self.w_u = nn.Linear(d_emb, d_emb)
        self.drop_u = nn.Dropout(p=dropout)
        self.ln_u = nn.Linear(d_emb, d_emb)

    def forward(self, x, cntx, cntx_mask=None):
        _bs, _, d = x.size()  # (b, n, d_emb)
        q, k, v = [
            lp(ln(t)).view(_bs, -1, self.h, self.d_h).transpose(1, 2)  # d_emb => h x d_h
            for lp, ln, t in zip(self.w_kqv, self.ln_kqv, [x, cntx, cntx])
        ]

        # Attention
        x_attn, attn = dot_product_attention(q, k, v, kv_mask=cntx_mask, drop=self.drop_attn)

        # Concat => transpose => project(dropout) => layer norm
        x = self.w_u(x_attn.transpose(1, 2).contiguous().view(_bs, -1, self.h * self.d_k))
        if self.drop_u is not None:
            x = self.drop_u(x)
        x = self.ln_u(x + x_attn)
        return x

class BertIntermediate(nn.Module):
    def __init__(self, d_emb, d_inter=4096, hidden_act=gelu):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(d_emb, d_inter)
        self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, d_emb, dropout=0.1):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(d_emb, d_emb)
        self.LayerNorm = BertLayerNorm(d_emb, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, d_emb, h, dropout):
        super().__init__()
        self.attention = BertAttentionLayer(d_emb, h, dropout)
        self.intermediate = BertIntermediate(d_emb)
        self.output = BertOutput(d_emb)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class MMBert(SimplifiedBert):
    """Stack of Self-Attention layers for multi-modal interaction"""
    def __init__(self, d_emb, **kwargs):
        super().__init__()
        self.attn_layers = _get_clones(
            BertLayer(d_emb, n_head, dropout), n_layer
        )
        # self.apply(self._init_weights())

    def prepare_output(self, x, out):
        iml = out['image']['embedding'].size(1)
        out['image']['embedding'] = x[:, 0, :]
        out['image']['cls_embedding'] = [:, 1:iml, :]
        out['section']['embedding'] = x[:, iml, :]
        out['section']['cls_embedding'] = x[:, iml+1:, :]

    def prepare_input(self, out):
        self._prepare_input(out)
        x = torch.cat([
            out['image']['cls_embedding'],
            out['image']['embedding'],
            out['section']['cls_embedding'],
            out['section']['embedding'],
        ], dim=1)
        _bs = x.size(0)
        x_mask = torch.cat([
            torch.ones((_bs, 1), device=x.device).long(),
            out['image']['mask'],
            torch.ones((_bs, 1), device=x.device).long(),
            out['section']['mask']
        ], dim=1).bool()
        return x, x_mask

    def forward(self, out, return_states=False):
        x, x_mask = self.prepare_input(out)
        all_states = []
        x_prev = x
        for layer in self.layers:
            x_prev = layer(x_prev, mask=x_mask)
            if return_states:
                all_states.append(x_prev)
        self.prepare_output(x_prev, out)
        if return_states:
            return out, all_states
        return out
