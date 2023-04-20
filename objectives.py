import numpy as np
import tqdm
import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from utils import pl_utils

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist

def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace

@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)  # this is b, (b, m)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)  # (b, n, m)
    A = torch.exp(-C.transpose(1, 2) / beta)  # this is G, (b, n, m)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)  # (b, m)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)  # (b, n, m)
    A.masked_fill_(joint_pad, 0)  

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)  # this is a, (b, n)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)  # this is a, (b, m)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T

def compute_lm(model, out):
    out = model.decode(out)
    # acc = (out['logits'][..., :-1, :].contiguous(), out['labels'][..., 1:].contiguous())
    ret = {
        'lm_loss': out['loss'],
        # 'lm_acc': acc,
    }
    return ret

def compute_wd(model, out):
    n_embed = model._config['n_embeds']
    wl = model._config['wd_lambda']
    im_emb = out['image']['residual']
    txt_emb = out['section']['residual']
    bs = im_emb.size(0)
    im_pad = torch.zeros((bs, n_embed), device=im_emb.device).bool()
    txt_pad = torch.zeros((bs, n_embed), device=txt_emb.device).bool()

    with torch.cuda.amp.autocast(enabled=False):
        cost = cost_matrix_cosine(txt_emb.float(), im_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | im_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        im_len = (im_pad.size(1) - im_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, im_len, im_pad, joint_pad, 0.5, 50, 1
        )
        dist = trace(cost.matmul(T.detach()))

    wd_loss = dist.mean()
    
    ret = {"wd_loss": wl * wd_loss}
    return ret

def compute_div(model, out):
    n_embed = model._config['n_embeds']
    dl = model._config['div_lambda']
    im_emb, txt_emb = out['image']['residual'], out['section']['residual']
    
    div_loss = diversity_loss(im_emb, n_embed) + diversity_loss(txt_emb, n_embed)
    
    ret = {"div_loss": dl * div_loss}
    return ret

def diversity_loss(x, n_embed):
    x = F.normalize(x, p=2, dim=-1)
    gram_x = x.bmm(x.transpose(1,2))
    I = (torch.eye(n_embed) > 0.5).repeat(gram_x.size(0), 1, 1).to(gram_x.device)
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (n_embed**2)
    return loss.mean()