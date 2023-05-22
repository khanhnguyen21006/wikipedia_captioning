import torch
import torch.nn.functional as F

import numpy as np

from utils import *

def cosine_distance(x, y, eps=1e-5):
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist

@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    b, m, n = C.size()  # C: (b, m, n)
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)  # this is b, (b, m)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)  # (b, n, m)
    A = torch.exp(-C.transpose(1, 2) / beta)  # this is G, (b, n, m) note: C transposed

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)  # (b, m)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)  # (b, n, m)
    A.masked_fill_(joint_pad, 0)  # (b, n, m)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # (b, n, m)
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)  # this is a, (b, n)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)  # this is b, (b, m)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T

def wasserstein_dist(x, y, n):
    bs = x.size(0)
    x_pad = torch.zeros((bs, n), device=x.device).bool()
    y_pad = torch.zeros((bs, n), device=y.device).bool()

    with torch.cuda.amp.autocast(enabled=False):
        cost = cosine_distance(x.float(), y.float())
        joint_pad = x_pad.unsqueeze(-1) | y_pad.unsqueeze(-2)
        cost.masked_fill_(joint_pad, 0)
        x_len = (x_pad.size(1) - x_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        y_len = (y_pad.size(1) - y_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), x_len, x_pad, y_len, y_pad, joint_pad, 0.5, 50, 1
        )
        dist = trace(cost.matmul(T.detach()))
    return dist.sum()

def diversity_loss(x, n):
    x = F.normalize(x, p=2, dim=-1)
    gram_x = x.bmm(x.transpose(1,2))
    I = (torch.eye(n) > 0.5).repeat(gram_x.size(0), 1, 1).to(gram_x.device)
    gram_x.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(g, p=2) for g in gram_x]) / (n**2)
    return loss.sum()

def radial_basis_kernel(x, y, gamma):
    pdist = torch.norm(x[:, None] - y, dim=2, p=2)
    return torch.exp(-gamma * pdist)

def mmd_rbf_loss(x, y, gamma=None):
    if gamma is None:
        gamma = 1./x.size(-1)
    loss = radial_basis_kernel(x, x, gamma) - 2*radial_basis_kernel(x, y, gamma) + radial_basis_kernel(y, y, gamma)
    return loss.sum()

def mil_loss(x, y, m=0.2, mine=False, return_score=False):
    assert len(x.size()) == len(y.size()) and x.size(0) == y.size(0)
    _bs = x.size(0)
    x = F.normalize(x, p=2, dim=-1)  # (b*b, n, d)
    y = F.normalize(y, p=2, dim=-1)  # (b*b, n, d)
    score = x.view(-1, x.size(-1)).mm(y.view(-1, y.size(-1)).t())
    score = model.max_pool(score.unsqueeze(0)).squeeze()
    if return_score:
        return score
    i2t, t2i = triplet_ranking_loss(score, m, mine)
    max_score = torch.max(score, dim=0).indices - torch.arange(0, _bs, device='cuda')
    recall = torch.count_nonzero(max_score == 0)/_bs
    return i2t, t2i, recall

def soft_contrastive_nll(logit, matched):
    if len(matched.size()) == 1:
        matched = matched[:, None]
    return -((logit * matched - torch.stack((logit, -logit), dim=2).logsumexp(dim=2, keepdim=False)).logsumexp(dim=1)) + np.log(logit.size(1))

def kl_divergence(mu, logsig):
    return -0.5 * (1 + logsig - mu.pow(2) - logsig.exp()).sum()

def prob_emb_loss(dist1, dist2, scale, shift):
    dist1, dist2, matched = pairwise_sampling(dist1, dist2)
    distance = batchwise_cosine_distance(dist1, dist2)

    logits = -scale * distance + shift

    idx = matched == 1
    loss_pos = soft_contrastive_nll(logits[idx], matched[idx]).sum()
    idx = matched != 1
    loss_neg = soft_contrastive_nll(logits[idx], matched[idx]).sum()

    ret = {
        'loss': loss_pos + loss_neg,
        'pos_loss': loss_pos,
        'neg_loss': loss_neg,
    }
    return ret

def compute_lm(model, out):
    out = model.decode(out)
    # acc = (out['logits'][..., :-1, :].contiguous(), out['labels'][..., 1:].contiguous())
    ret = {
        'lm_loss': out['loss'],
        # 'lm_acc': acc,
    }
    return ret

def compute_wd(model, out):
    n_emb = model._config['n_embed']
    wl = model._config['wd_lambda']
    image_emb = out['image']['residual']
    text_emb = out['section']['residual']
    wd_loss = wasserstein_dist(text_emb, image_emb, n_emb)
    ret = {"wd_loss": wl * wd_loss}
    return ret

def compute_div(model, out):
    n_emb, dl = model._config['n_embed'], model._config['div_lambda']
    assert n_emb > 1
    image_emb, text_emb = out['image']['residual'], out['section']['residual']
    div_loss = diversity_loss(image_emb, n_emb) + diversity_loss(text_emb, n_emb)
    ret = {"div_loss": dl * div_loss}
    return ret

def compute_mmd(model, out):
    n_emb, ml = model._config['n_embed'], model._config['mmd_lambda']
    assert n_emb > 1
    image_emb, text_emb = out['image']['residual'], out['section']['residual']
    mmd_loss = mmd_rbf_loss(image_emb.view(-1, image_emb.size(-1)), text_emb.view(-1, text_emb.size(-1)), gamma=0.5)
    ret = {"mmd_loss": ml * mmd_loss}
    return ret

def compute_de(model, out):
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    assert n_emb == 1 and not prob_emb and (mm_query is None and len(source) == 1) or (mm_query is not None and len(source) == 2)
    if mm_query is None:
        image_emb, text_emb = out[source[0]]['embedding'].unsqueeze(1), out[target]['embedding'].unsqueeze(1)
    elif mm_query == 'addition':
        image_emb = out['image']['embedding'].unsqueeze(1) + out[source[1]]['embedding'].unsqueeze(1)
        text_emb = out[target]['embedding'].unsqueeze(1)
    elif mm_query == 'average':
        image_emb = (out['image']['embedding'].unsqueeze(1) + out[source[1]]['embedding'].unsqueeze(1))/2
        text_emb = out[target]['embedding'].unsqueeze(1) 
    else:
        raise ValueError("Invalid query composition.")
    i2t = prob_emb_loss(image_emb, text_emb, model.scale, model.shift)
    t2i = prob_emb_loss(text_emb, image_emb, model.scale, model.shift)
    ret = {
        'i2t': i2t['loss'].item(),
        't2i': t2i['loss'].item(),
        'i2t_pos': i2t['pos_loss'].item(),
        'i2t_neg': i2t['neg_loss'].item(),
        't2i_pos': t2i['pos_loss'].item(),
        't2i_neg': t2i['neg_loss'].item(),
        'de_loss': (i2t['loss'] + t2i['loss'])/2,
    }
    return ret

def compute_pe(model, out):
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    assert n_emb > 1 and prob_emb and ((mm_query is None and len(source) == 1) or (mm_query is not None and len(source) == 2))
    if mm_query is None:
        image_emb, text_emb = out[source[0]]['embedding'], out[target]['embedding']
    else:
        # for PE+addition: average samples from distibution to compute the mean
        im_mu, im_logsig = out['image']['embedding'].mean(dim=1), out['image']['logsigma']
        txt_mu, txt_logsig = out[source[1]]['embedding'].mean(dim=1), out[source[1]]['logsigma']
        if mm_query == 'addition':
            mu, logsig, _ = addition_2_gaussians(im_mu, im_logsig, txt_mu, txt_logsig)
        elif mm_query == 'mixture':
            mu, logsig, _ = mixture_2_gaussians(im_mu, im_logsig, txt_mu, txt_logsig)
        else:
            raise ValueError("Invalid query composition.")
        image_emb = sample_gaussian_tensors(mu, logsig, n_emb)
        text_emb = out[target]['embedding']
    i2t = prob_emb_loss(image_emb, text_emb, model.scale, model.shift)
    t2i = prob_emb_loss(text_emb, image_emb, model.scale, model.shift)
    ret = {
        'i2t': i2t['loss'].item(),
        't2i': t2i['loss'].item(),
        'i2t_pos': i2t['pos_loss'].item(),
        'i2t_neg': i2t['neg_loss'].item(),
        't2i_pos': t2i['pos_loss'].item(),
        't2i_neg': t2i['neg_loss'].item(),
        'pe_loss': (i2t['loss'] + t2i['loss'])/2,
    }
    return ret

def compute_tripe(model, out):
    vl = model._config['vib_lambda']
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    assert n_emb > 1 and prob_emb and ((mm_query is None and len(source) == 2))

    image_emb, image_logsig = out['image']['embedding'], out['image']['logsigma']
    source1_emb, source1_logsig = out[source[1]]['embedding'], out[source[1]]['logsigma']
    text_emb, text_logsig = out[target]['embedding'], out[target]['logsigma']
    i2s = prob_emb_loss(image_emb, text_emb, model.si_scale, model.si_shift)
    s2i = prob_emb_loss(text_emb, image_emb, model.si_scale, model.si_shift)
    s2d = prob_emb_loss(text_emb, source1_emb, model.ds_scale, model.ds_shift)
    d2s = prob_emb_loss(source1_emb, text_emb, model.ds_scale, model.ds_shift)
    i2d = prob_emb_loss(image_emb, source1_emb, model.id_scale, model.id_shift)
    d2i = prob_emb_loss(source1_emb, image_emb, model.id_scale, model.id_shift)
    kl_image = kl_divergence(image_emb.mean(dim=1), image_logsig)
    kl_source1 = kl_divergence(source1_emb.mean(dim=1), source1_logsig)
    kl_text = kl_divergence(text_emb.mean(dim=1), text_logsig)

    ret = {
        'i2s': i2s['loss'].item(),
        's2i': s2i['loss'].item(),
        'd2s': d2s['loss'].item(),
        's2d': s2d['loss'].item(),
        'i2d': i2d['loss'].item(),
        'd2i': d2i['loss'].item(),
        'i2s_pos': i2s['pos_loss'].item(),
        'i2s_neg': i2s['neg_loss'].item(),
        's2i_pos': s2i['pos_loss'].item(),
        's2i_neg': s2i['neg_loss'].item(),
        'd2s_pos': d2s['pos_loss'].item(),
        'd2s_neg': d2s['neg_loss'].item(),
        's2d_pos': s2d['pos_loss'].item(),
        's2d_neg': s2d['neg_loss'].item(),
        'i2d_pos': i2d['pos_loss'].item(),
        'i2d_neg': i2d['neg_loss'].item(),
        'd2i_pos': d2i['pos_loss'].item(),
        'd2i_neg': d2i['neg_loss'].item(),
        'tripe_loss': ((i2s['loss'] + s2i['loss']+ i2d['loss'] + d2i['loss'])*2 + d2s['loss'] + s2d['loss'])/10,
        'vib_loss': vl * (kl_image + kl_source1 + kl_text)/3,
        'image_volume': torch.exp(torch.mean(image_logsig)).item(),
        'description_volume': torch.exp(torch.mean(source1_logsig)).item(),
        'section_volume': torch.exp(torch.mean(text_logsig)).item(),
    }
    return ret

def compute_mmpe(mopdel, out):
    l2l = model._config['mmpe_l2_lambda']
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    assert n_emb > 1 and prob_emb and mm_query is not None and len(source) == 2
    # for MMPE: use only 1 sample as the mean of gaussian 
    im_mu, im_logsig = out['image']['embedding'], out['image']['logsigma']
    txt_mu, txt_logsig = out[source[1]]['embedding'], out[source[1]]['logsigma']
    mu, logsig, log_z = product_2_gaussians(im_mu, im_logsig, txt_mu, txt_logsig)
    text_mu, text_logsig = out[target]['embedding'], out[target]['logsigma']

    i2t, recall, _ = mmpe_loss(mu, logsig, log_z, text_mu, text_logsig, torch.zeros_like(log_z), n=n_emb, recall=True)
    t2i = mmpe_loss(text_mu, text_logsig, torch.zeros_like(log_z), mu, logsig, log_z, n=n_emb)
    logsig_l2_loss = torch.mean(torch.square(im_logsig)) + torch.mean(torch.square(txt_logsig)) + torch.mean(torch.square(text_logsigm))
    ret = {
        'i2t': i2t.item(),
        't2i': t2i.item(),
        'mmpe_loss': (i2t + t2i)/2,
        'logsig_l2_loss': (l2l/3) * logsig_l2_loss,
        'r@1_per_batch': recall.item(),
    }
    return ret

def compute_vib(model, out):
    vl = model._config['vib_lambda']
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    assert n_emb > 1 and prob_emb and ((mm_query is None and len(source) == 1) or (mm_query is not None and len(source) == 2))
    if mm_query is None:
        image_mu, image_logsig = out[source[0]]['embedding'].mean(dim=1), out[source[0]]['logsigma']
    else:
        im_mu, im_logsig = out['image']['embedding'].mean(dim=1), out['image']['logsigma']
        txt_mu, txt_logsig = out[source[1]]['embedding'].mean(dim=1), out[source[1]]['logsigma']
        if mm_query == 'addition':
            image_mu, image_logsig, _ = addition_2_gaussians(im_mu, im_logsig, txt_mu, txt_logsig)
        elif mm_query == 'mixture':
            image_mu, image_logsig, _ = mixture_2_gaussians(im_mu, im_logsig, txt_mu, txt_logsig)
        else:
            raise ValueError("Invalid query composition.")
    text_mu, text_logsig = out[target]['embedding'].mean(dim=1), out[target]['logsigma']
    vib_loss = kl_divergence(image_mu, image_logsig) + kl_divergence(text_mu, text_logsig)
    ret = {
        'vib_loss': vl * vib_loss,
        'image_volume': torch.exp(torch.mean(image_logsig)).item(),
        'text_volume': torch.exp(torch.mean(text_logsig)).item(),
    }
    return ret

def compute_se(model, out):
    n_emb, prob_emb, mm_query = model._config['n_embed'], model._config['prob_embed'], model._config['multi_query']
    source, target = model._config['source_to_target']['source'], model._config['source_to_target']['target']
    margin,  hard_mining = model._config['margin'], model._config['hard_mining']
    assert n_emb > 1 and not prob_emb and mm_query is None
    image_emb, text_emb = out[source[0]]['embedding'], out[target]['embedding']
    if model._config['se_match'] == 'smooth_chamfer':
        alpha = model._config['chamfer_alpha']
        i2t, t2i, recall = smooth_chamfer_loss(image_emb, text_emb, alpha, margin, hard_mining)
    elif model._config['se_match'] == 'multi_instance':
        i2t, t2i, recall = mil_loss(image_emb, text_emb, margin, hard_mining)
    else:
        raise ValueError(f"Invalid Set Embedding Matching: {model._config['se_match']}")
    ret = {
        'i2t': i2t.item(),
        't2i': t2i.item(),
        'r@1_per_batch': recall.item(),
        'se_loss': (i2t + t2i)/2,
    }
    return ret