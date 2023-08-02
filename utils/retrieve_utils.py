import os, re
import tqdm
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

from .tensor_utils import *

EMBED_OUT = [
    'image_emb', 'image_logsig', 'text_emb', 'text_logsig',
    'im_emb', 'im_logsig', 'txt_emb', 'txt_logsig',
    'image_z', 'text_z'
]

def rank(matrix, n, refs=None, return_rank=False):
    rank = np.zeros(n)
    top1 = np.zeros(n)
    rPrec = np.zeros(n)
    for idx in range(n):
        d = matrix[idx]
        inds = np.argsort(d)[::-1]
        if refs is not None:
            assert len(refs) == n
            _rank = 1e20
            K = len(refs[idx])
            for ref_idx in refs[idx]:
                r = np.where(inds == ref_idx)[0][0]
                if r < _rank:
                    _rank = r
            rP = len(set(inds[:K]).intersection(set(refs[idx]))) / len(set(refs[idx]))
        else:
            rP = 0
            _rank = np.where(inds == idx)[0][0]
        rPrec[idx] = rP
        rank[idx] = _rank
        top1[idx] = inds[0]
    r1 = 100.0 * len(np.where(rank < 1)[0]) / len(rank)
    r5 = 100.0 * len(np.where(rank < 5)[0]) / len(rank)
    r10 = 100.0 * len(np.where(rank < 10)[0]) / len(rank)
    rPrec = 100.0 * np.mean(rPrec)
    medr = np.floor(np.median(rank)) + 1
    meanr = rank.mean() + 1
    if return_rank:
        return {
            "R@1": r1, "R@5": r5, "R@10": r10, "R-Precision": rPrec,
            "Medium Rank": medr, "Mean Rank": meanr,
        }, (rank, top1)
    else:
        return {
            "R@1": r1, "R@5": r5, "R@10": r10, "R-Precision": rPrec, 
            "Medium Rank": medr, "Mean Rank": meanr,
        }

def match_prob(dist1, dist2, scale, shift):
    distance = batchwise_cosine_distance(dist1, dist2).to(scale.device).float()
    logits = -scale * distance + shift
    prob = torch.exp(logits) / (torch.exp(logits) + torch.exp(-logits))
    return prob.mean(axis=1)

def triplet_ranking_loss(S, m=0.2, mine=False):
    diagonal = S.diag().view(S.size(0), 1)
    d1 = diagonal.expand_as(S)
    d2 = diagonal.t().expand_as(S)

    mask = torch.eye(S.size(0)) > .5
    I = torch.autograd.Variable(mask).cuda()

    l1, l2 = (m + S - d1).clamp(min=0.0), (m + S - d2).clamp(min=0.0)
    l1.masked_fill_(I, 0.0)
    l2.masked_fill_(I, 0.0)
    if mine:
        l1, l2 = l1.max(0)[0], l2.max(1)[0]
    return l1.sum(), l2.sum()

def smooth_chamfer_loss(x, y, alpha, m=0.2, mine=False, return_score=False):
    assert len(x.size()) == len(y.size()) and x.size(0) == y.size(0)
    _bs = x.size(0)
    x_sampled, y_sampled, _ = pairwise_sampling(x, y)
    x = F.normalize(x_sampled, p=2, dim=-1)  # (b*b, n, d)
    y = F.normalize(y_sampled, p=2, dim=-1)  # (b*b, n, d)

    cosim = x.bmm(y.transpose(1, 2))  # (b*b, n, n)
    x_score = 1/(2*alpha) * torch.mean(torch.logsumexp(alpha * cosim, dim=-1), dim=-1)  # (b*b)
    y_score = 1/(2*alpha) * torch.mean(torch.logsumexp(alpha * cosim, dim=1), dim=-1)  # (b*b)
    score = (x_score + y_score).view(_bs, _bs)  # (b, b)

    if return_score:
        return score
    i2t, t2i = triplet_ranking_loss(score, m, mine)
    max_score = torch.max(score, dim=0).indices - torch.arange(0, _bs, device='cuda')
    recall = torch.count_nonzero(max_score == 0)/_bs
    return i2t, t2i, recall

def mmpe_loss(mu1, logsig1, z1, mu2, logsig2, z2, n=7, reduction='mean', recall=False):
    bs = mu1.size(0)
    samples = sample_gaussian_tensors(mu2, logsig2, n).cuda()

    # likelihood derivation
    # https://stats.stackexchange.com/questions/351549/maximum-likelihood-estimators-multivariate-gaussian
    inv_sigmas = torch.exp(-logsig1)
    loc = -0.5 * torch.mean(torch.sum(((samples.unsqueeze(0) - mu1.unsqueeze(1).unsqueeze(2)) ** 2) * inv_sigmas.unsqueeze(1).unsqueeze(2), dim=-1), dim=-1)
    norm = (-mu1.shape[-1]/2) * torch.log(torch.Tensor([2*math.pi]).cuda()) - 0.5 * (torch.sum(logsig1, dim=-1))
    scores = z1 + norm + loc
    logits = scores
    scores = scores - torch.max(scores, dim=0, keepdim=True).values

    labels = torch.arange(0, mu1.shape[0]).long().to(scores.get_device())
    loss = F.cross_entropy(scores, labels, reduction=reduction)
    if recall:
        max_scores = torch.max(scores, dim=0).indices - torch.arange(0, mu1.shape[0], device='cuda')
        recall = torch.count_nonzero(max_scores == 0) / bs
        return loss, recall, F.softmax(logits, dim=-1)
    return loss

def ntxent_loss(x, y, mat=None, x_mask=None, y_mask=None, T=0.1, reduction='mean'):
    """
        InfoNCE loss for cross-modality matching, following https://arxiv.org/abs/1807.03748v2.
        Negatives are implicitly off-diagonal positives.
        x: (b, d) or (b, n, d)
        y: (b, d) or (b, m, d)
        x_mask: (b, n), y_mask: (b, m)
        1. compute cosine matrix
        2. compute numerator (this is repeated from 1. but escape passing seperate pos/neg tensors)
        3. compute denomerator based on 1.
        4. compute loss
    """
    if len(x) != len(y) or x.size(-1) != y.size(-1):
        raise ValueError('x and y must have same first and last dimension.')
    if len(x.size()) not in [2, 3] or len(y.size()) not in [2, 3]:
        raise ValueError('Only support 2or3-dimension tensor.')

    # Assume all the tensors are l2 normalized
    if len(x.size()) == 3 and len(y.size()) == 3:
        assert x_mask != None and x_mask.size() == x.size()[:2]
        assert y_mask != None and y_mask.size() == y.size()[:2]
        if mat == None:
            mat = x.reshape(-1, x.size(-1)) @ y.reshape(-1, y.size(-1)).transpose(-2, -1)  # (b*n, b*m)

        # mat1 = mat.masked_fill_(~y_mask.flatten(), -np.inf)  # (b*n, b*m)
        # pos1 = torch.bmm((x, y.transpose(-2, -1)), dim=2)  # (b, n, m)
        pos1_mask = torch.block_diag(*y_mask)  # (b, b*m)
        pos1 = mat.masked_fill(~pos1_mask, -np.inf)  # (b*n, b*m)
        neg1_mask = y_mask.flatten()  # (b*m)
        neg1 = mat.masked_fill(~neg1_mask, -np.inf)  # (b*n, b*m)

        pos1 = torch.logsumexp(pos1[x_mask] - torch.max(pos1[x_mask], dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (some dim0)
        neg1 = torch.logsumexp(neg1[x_mask] - torch.max(neg1[x_mask], dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (some dim0)
        loss1 = (neg1 - pos1).mean()

        pos2_mask = torch.block_diag(*x_mask).transpose(0, 1)  # (b*n, b)
        pos2 = mat.masked_fill(~pos2_mask, -np.inf)  # (b*n, b*m)
        neg2_mask = x_mask.flatten().transpose(0, 1)  # (b*n)
        neg2 = mat.masked_fill(~neg2_mask, -np.inf)  # (b*n, b*m)

        pos2 = torch.logsumexp(pos2[y_mask] - torch.max(pos2[y_mask], dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (some dim1)
        neg2 = torch.logsumexp(neg2[y_mask] - torch.max(neg2[y_mask], dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (some dim1)
        loss2 = (neg2 - pos2).mean()

        loss = (loss1 + loss2)/2
    elif len(x.size()) == 2 and len(y.size()) == 3:
        assert y_mask != None and y_mask.size() == y.size()[:2]
        if mat == None:
            mat = x @ y.reshape(-1, y.size(-1)).transpose(0, 1)  # (b, b*m)

        # pos = torch.sum((x[:, None, :] * y), dim=2)  # (b, m)
        pos_mask = torch.block_diag(*y_mask)  # (b, b*m)
        pos = mat.masked_fill(~pos_mask, -np.inf)  # (b, b*m)
        neg_mask = y_mask.flatten()  # (b*m)
        neg = mat.masked_fill(~neg_mask, -np.inf)  # (b, b*m)

        pos = torch.logsumexp(pos - torch.max(pos, dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (b)
        neg = torch.logsumexp(neg - torch.max(neg, dim=-1)[0].unsqueeze(1).detach(), dim=-1)  # (b)
        loss1 = (neg - pos).mean()

        logit = mat[:, y_mask.flatten()].transpose(0, 1)  # (some dim ,b)
        label = torch.arange(len(y), device=y.device).unsqueeze(1).expand_as(y_mask)[y_mask].flatten()  # (some dim)
        loss2 = F.cross_entropy(logit, label, reduction=reduction)

        loss = (loss1 + loss2)/2
    elif len(x.size()) == 2 and len(y.size()) == 2:
        logit = x @ y.transpose(-2, -1)  if mat == None else mat # (b, b)
        label = torch.arange(len(x), device=x.device)
        loss = (F.cross_entropy(logit/T, label, reduction=reduction) + 
                    F.cross_entropy(logit.transpose(0, 1)/T, label, reduction=reduction))/2
    else:
        raise ValueError('Invalid combination.')
    return loss

@torch.no_grad()
def retrieve_wrapup(pl_module):
    """
	1. call dataloaders
	2. loop over valset -> extract features per batch
	3. gather distributedly-computed features
	4. compute score
    """
    _config = pl_module.hparams._config
    test_split = _config['retrieval_testset']
    if 'multi' in test_split:
        retrieve_multi(pl_module)
        return

    ws = torch.distributed.get_world_size()

    ret_dset= pl_module.trainer.datamodule.get_ret_dset(test_split)
    ret_dset.hparams = pl_module.trainer.datamodule.collate_hparams
    dist_sampler = DistributedSampler(ret_dset, shuffle=False)
    ret_loader = torch.utils.data.DataLoader(
        ret_dset,
        batch_size=_config["per_gpu_batchsize"],
        num_workers=_config["num_workers"],
        pin_memory=True,
        collate_fn=partial(ret_dset.collate),
        sampler=dist_sampler,
    )
    n_query = len(ret_dset)
    n_emb, prob_emb, mm_query = _config["n_embed"], _config['prob_embed'], _config['multi_query']
    source, target = _config['source_to_target']['source'], _config['source_to_target']['target']
    emb_out = {k: list() for k in EMBED_OUT}
    for _b in tqdm.tqdm(ret_loader, desc="test retrieval loop"):
        _b = {_k: (_v.to(pl_module.device) if isinstance(_v, torch.Tensor) else _v) for _k, _v in _b.items()}
        model_out = pl_module.model(_b)
        if n_emb > 1:
            if prob_emb:
                update_prob_embed(emb_out, model_out, mm_query, source, target, n_emb)
            else:
                update_set_embed(emb_out, model_out, source, target)
        elif n_emb == 1:
            update_det_embed(emb_out, 'embedding', model_out, mm_query, source, target)
        else:
            update_det_embed(emb_out, 'cls_embedding', model_out, mm_query, source, target)
    emb_out = {k: torch.cat(v, dim=0) for k, v in emb_out.items() if len(v) > 0}

    torch.distributed.barrier()

    buffer = {k: distributed_all_gather(v) for k, v in emb_out.items()}
    ret_matrix = torch.zeros(n_query, n_query)
    if _config['eval_method'] == 'matmul':
        for idx, query in enumerate(buffer['image_emb']):
            query = query.unsqueeze(0)
            query = query.view(len(query) * n_emb, -1)
            gallery = buffer['text_emb'].view(n_query * n_emb, -1).t()
            _score = query.mm(gallery)
            if n_emb > 1:
                _score = _score.view(
                    len(query)//n_emb, n_emb, gallery.size()[-1]//n_emb, n_emb
                )
                _score = _score.permute(0, 1, 3, 2)
                _score = torch.sum(torch.sum(_score, axis=1), axis=1)
            ret_matrix[idx] = _score
    elif _config['eval_method'] == 'matching_prob':
        scale, shift = pl_module.model.scale, pl_module.model.shift
        for idx, query in enumerate(buffer['image_emb']):
            _score = match_prob(query.unsqueeze(0), buffer['text_emb'], scale, shift)
            ret_matrix[idx] = _score
    elif _config['eval_method'] == 'mmpe_loss':
        assert mm_query == 'multiplication'
        for idx in range(buffer['image_emb'].size(0)):
            query_mu = buffer['image_emb'][idx].unsqueeze(0)
            query_logsig = buffer['image_logsig'][idx].unsqueeze(0)
            query_z = buffer['image_z'][idx].unsqueeze(0)
            _, _ , ret_matrix[idx] = mmpe_loss(
                    query_mu, query_logsig, query_z, 
                    tgt_means, tgt_sigs, torch.zeros_like(query_z), 
                    recall=True
                )
    elif _config['eval_method'] == 'mmpe_matmul':
        assert mm_query == 'multiplication'
        ret_matrix = buffer['image_emb'] @ buffer['text_emb'].t()
    elif _config['eval_method'] == 'multi_instance':
        assert n_emb > 1
        image_emb = F.normalize(buffer['image_emb'], dim=-1)
        text_emb = F.normalize(buffer['text_emb'], dim=-1)
        cosine_kk = image_emb.view(-1, image_emb.size(-1)).mm(
                text_emb.view(-1, text_emb.size(-1)).t()
            )  # (b*n_emb, b*n_emb)
        cosine_kk = cosine_kk.view(n_query, n_emb, n_query, n_emb)
        cosine_kk = cosine_kk.permute(0, 1, 3, 2).contiguous()
        cosine_kk = cosine_kk.view(n_query, -1, n_query)
        ret_matrix, _ = cosine_kk.max(dim=1)
    elif _config['eval_method'] == 'smooth_chamfer':
        assert n_emb > 1
        alpha = _config['chamfer_alpha']
        buffer['image_emb'] = F.normalize(buffer['image_emb'], dim=-1)
        buffer['text_emb'] = F.normalize(buffer['text_emb'], dim=-1)
        # import pudb; pu.db
        for idx in range(buffer['image_emb'].size(0)):
            x, y = image_emb[idx].unsqueeze(0).expand_as(text_emb), text_emb
            cosim = x.bmm(y.transpose(1, 2))  # (b, n, n)
            x_score = 1/(2*alpha) * torch.mean(torch.logsumexp(alpha * cosim, dim=-1), dim=-1)  # (b)
            y_score = 1/(2*alpha) * torch.mean(torch.logsumexp(alpha * cosim, dim=1), dim=-1)  # (b)
            ret_matrix[idx] = (x_score + y_score).view(1, text_emb.size(0))  # (1, b)
    elif 'mse_space' in _config['eval_method']:
        s = int(_config['eval_method'].split('_')[-1])
        assert s < n_emb, f"Invalid evaluation method: {_config['eval_method']}."
        ret_matrix = buffer['image_emb'][:, s, :] @ buffer['text_emb'][:, s, :].t()
    elif _config['eval_method'] == 'clip':
        assert n_emb == 0 and mm_query == None
        image_emb = buffer['image_emb'] / buffer['image_emb'].norm(dim=-1, keepdim=True)
        text_emb = buffer['text_emb'] / buffer['text_emb'].norm(dim=-1, keepdim=True)
        image_emb, text_emb = image_emb.squeeze(1), text_emb.squeeze(1)
        ret_matrix = torch.matmul(image_emb, text_emb.t()) * pl_module.model.image_encoder.logit_scale.exp()
    else:
        raise ValueError(f"Method {_config['eval_method']} is not supported for evaluation.")

    ret_matrix = ret_matrix.detach().cpu().numpy()
    i2t = rank(ret_matrix, n_query)
    t2i = rank(np.transpose(ret_matrix), n_query)
    print("i2t retrieval: ", i2t)
    print("t2i retrieval: ", t2i)

    if torch.distributed.get_rank() == 0:
        expt_path = os.path.join(
            _config['result_dir'], 'inference', _config['expt_name'], 'retrieve'
        )
        os.makedirs(expt_path, exist_ok=True)
        print(f"Saving retrieval output to: {expt_path}")

        if not n_emb > 1 or not prob_emb:
            if mm_query is None:
                np.save(os.path.join(expt_path, f"image_to_{target}_score_{test_split}"), ret_matrix)
            else:
                np.save(os.path.join(expt_path, f"image_{source[1]}_to_{target}_score_{test_split}"), ret_matrix)
            return

        if mm_query is None:
            np.save(os.path.join(expt_path, f"image_to_{target}_score_{test_split}"), ret_matrix)
            np.save(os.path.join(expt_path, f"image_embedding_{test_split}"), buffer['image_emb'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"image_logsigma_{test_split}"), buffer['image_logsig'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"{target}_embedding_{test_split}"), buffer['text_emb'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"{target}_logsigma_{test_split}"), buffer['text_logsig'].detach().cpu().numpy())
        else:
            np.save(os.path.join(expt_path, f"image_{source[1]}_to_{target}_score_{test_split}"), ret_matrix)
            np.save(os.path.join(expt_path, f"image_logsigma_{test_split}"), buffer['image_logsig'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"im_logsigma_{test_split}"), buffer['im_logsig'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"{source[1]}_logsigma_{test_split}"), buffer['txt_logsig'].detach().cpu().numpy())
            np.save(os.path.join(expt_path, f"{target}_logsigma_{test_split}"), buffer['text_logsig'].detach().cpu().numpy())
            if mm_query == 'addition' or 'mixture':
                np.save(os.path.join(expt_path, f"image_embedding_{test_split}"), buffer['image_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"im_embedding_{test_split}"), buffer['im_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"{source[1]}_embedding_{test_split}"), buffer['txt_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"{target}_embedding_{test_split}"), buffer['text_emb'].detach().cpu().numpy())
            elif mm_query == 'multiplication':
                np.save(os.path.join(expt_path, f"image_mu_{test_split}"), buffer['image_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"im_mu_{test_split}"), buffer['im_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"{source[1]}_mu_{test_split}"), buffer['txt_emb'].detach().cpu().numpy())
                np.save(os.path.join(expt_path, f"{target}_mu_{test_split}"), buffer['text_emb'].detach().cpu().numpy())

def update_prob_embed(emb_out, model_out, mm_query, source, target, n_emb):
    _image_emb, _text_emb = model_out[source[0]]['embedding'], model_out[target]['embedding']
    _image_logsig, _text_logsig = model_out[source[0]]['logsigma'], model_out[target]['logsigma']
    if mm_query is not None:
        _im_emb, _im_logsig = model_out['image']['embedding'], model_out['image']['logsigma']
        _txt_emb, _txt_logsig = model_out[source[1]]['embedding'], model_out[source[1]]['logsigma']
        if mm_query == 'addition':
            _im_mu, _txt_mu = _im_emb.mean(dim=1), _txt_emb.mean(dim=1)
            _image_mu, _image_logsig, _ = addition_2_gaussians(_im_mu, _im_logsig, _txt_mu, _txt_logsig)
            _image_emb = sample_gaussian_tensors(_image_mu, _image_logsig, n_emb)
        if mm_query == 'mixture':
            _im_mu, _txt_mu = _im_emb.mean(dim=1), _txt_emb.mean(dim=1)
            _image_mu, _image_logsig, _ = mixture_2_gaussians(_im_mu, _im_logsig, _txt_mu, _txt_logsig)
            _image_emb = sample_gaussian_tensors(_image_mu, _image_logsig, n_emb)
        if mm_query == 'multiplication':
            _image_mu, _image_logsig, log_z = product_2_gaussians(_im_emb, _im_logsig, _txt_emb, _txt_logsig)
            _image_emb = _image_mu
            emb_out['image_z'].append(_im_emb)
        emb_out['im_emb'].append(_im_emb)
        emb_out['txt_emb'].append(_txt_emb)
        emb_out['im_logsig'].append(_im_logsig)
        emb_out['txt_logsig'].append(_txt_logsig)
    emb_out['image_emb'].append(_image_emb)
    emb_out['text_emb'].append(_text_emb)
    emb_out['image_logsig'].append(_image_logsig)
    emb_out['text_logsig'].append(_text_logsig)

def update_set_embed(emb_out, model_out, source, target):
    emb_out['image_emb'].append(model_out[source[0]]['embedding'])
    emb_out['text_emb'].append(model_out[target]['embedding'])

def update_det_embed(emb_out, key, model_out, mm_query, source, target):
    _image_emb, _text_emb = model_out[source[0]][key].unsqueeze(1), model_out[target][key].unsqueeze(1)
    if mm_query is not None:
        _im_emb, _txt_emb = model_out['image'][key].unsqueeze(1), model_out[source[1]][key].unsqueeze(1)
        if mm_query == 'addition':
            _image_emb = _im_emb + _txt_emb
        if mm_query == 'average':
            _image_emb = (_im_emb + _txt_emb)/2
        emb_out['im_emb'].append(_im_emb)
        emb_out['txt_emb'].append(_txt_emb)
    emb_out['image_emb'].append(_image_emb)
    emb_out['text_emb'].append(_text_emb)

@torch.no_grad()
def retrieve_multi(pl_module):
    """
    This function only supports retrieval usecases:
        image-to-section, image-to-caption
        image-description-to-section, image-description-to-caption, image-section-to-caption 
        output: image-to-text 2-way recall, R-Precision, the other way if possible
    """
    _config = pl_module.hparams._config
    dm = pl_module.trainer.datamodule
    dm.dataset = 'witretmulti'
    n_emb, prob_emb, mm_query, em = _config["n_embed"], _config['prob_embed'], _config['multi_query'], _config['eval_method']
    source, target, test_split = _config['source_to_target']['source'], _config['source_to_target']['target'], _config['retrieval_testset']
    assert re.match(r"match_sentence_(max|first)_(\d+)", em)

    if 'max' in em or 'first' in em:
        ext, wd = em.split('_')[-2], em.split('_')[-1]
    else:
        ext, wd = '', ''

    if 'test_1k' in _config['retrieval_testset']:
        txt_split = 'test_1k_RET_Sec' + f'_{ext}_{wd}'
        im_split = 'test_1k_RET_Im' + f'_{ext}_{wd}'
    elif'test_5k' in _config['retrieval_testset']:
        txt_split = 'test_5k_RET_Sec' + f'_{ext}_{wd}'
        im_split = 'test_5k_RET_Im' + f'_{ext}_{wd}'
    else:
        txt_split = 'test_SecDeDup_RET' if target == 'section' else 'test_CapDeDup_RET'
        im_split = 'test_ImDeDup_RET'

    txt_dset = dm.get_ret_dset(txt_split)
    txt_dset.hparams = dm.collate_hparams
    n_txt = len(txt_dset)
    txt_loader = torch.utils.data.DataLoader(
        txt_dset,
        batch_size=64,
        num_workers=_config["num_workers"],
        pin_memory=True,
        collate_fn=partial(txt_dset.collate),
    )

    im_dset = dm.get_ret_dset(im_split)
    im_dset.hparams = dm.collate_hparams
    n_im = len(im_dset)
    im_loader = torch.utils.data.DataLoader(
        im_dset,
        batch_size=_config["per_gpu_batchsize"],
        num_workers=_config["num_workers"],
        pin_memory=True,
        collate_fn=partial(im_dset.collate),
    )

    txt_embed, txt_logsig, txt_gt = list(), list(), list()
    for _b in tqdm.tqdm(txt_loader, desc='embedding text loop'):
        _b = {_k: (_v.to(pl_module.device) if isinstance(_v, torch.Tensor) else _v) for _k, _v in _b.items()}
        txt_out = pl_module.model._encode_text(_b, 'section')
        
        if n_emb > 1 and prob_emb:
            txt_embed.append(txt_out['embedding'])
            txt_logsig.append(txt_out['logsigma'])
        elif n_emb == 1:
            txt_embed.append(txt_out['embedding'])
        else:
            txt_embed.append(txt_out['cls_embedding'])
        txt_gt += _b['retrieve_gt']

    txt_embed = torch.cat(txt_embed, dim=0)
    if n_emb > 1 and prob_emb:
        txt_logsig = torch.cat(txt_logsig, dim=0)

    im_embed, im_logsig, im_gt = list(), list(), list()
    for _b in tqdm.tqdm(im_loader, desc='embedding image loop'):
        _b = {_k: (_v.to(pl_module.device) if isinstance(_v, torch.Tensor) else _v) for _k, _v in _b.items()}
        img_out = pl_module.model._encode_image(_b)

        if n_emb > 1:
            if prob_emb: # PE
                if mm_query is not None:
                    txt_out = pl_module.model._encode_text(_b, 'description')
                    _im_emb, _im_logsig = img_out['embedding'], img_out['logsigma']
                    _txt_emb, _txt_logsig = txt_out['embedding'], txt_out['logsigma']
                    if mm_query == 'addition':
                        _im_mu, _txt_mu = _im_emb.mean(dim=1), _txt_emb.mean(dim=1)
                        _image_mu, _image_logsig, _ = addition_2_gaussians(_im_mu, _im_logsig, _txt_mu, _txt_logsig)
                        _image_emb = sample_gaussian_tensors(_image_mu, _image_logsig, n_emb)
                    if mm_query == 'mixture':
                        _im_mu, _txt_mu = _im_emb.mean(dim=1), _txt_emb.mean(dim=1)
                        _image_mu, _image_logsig, _ = mixture_2_gaussians(_im_mu, _im_logsig, _txt_mu, _txt_logsig)
                        _image_emb = sample_gaussian_tensors(_image_mu, _image_logsig, n_emb)
                    if mm_query == 'multiplication':
                        _image_mu, _image_logsig, log_z = product_2_gaussians(_im_emb, _im_logsig, _txt_emb, _txt_logsig)
                        _image_emb = _image_mu
                    im_embed.append(_image_emb)
                    im_logsig.append(_image_logsig)
                else:
                    im_embed.append(img_out['embedding'])
                    im_logsig.append(img_out['logsigma'])
            else: # SE
                im_embed.append(img_out['embedding'])
        elif n_emb == 1: # DE
            if mm_query is not None:
                txt_out = pl_module.model._encode_text(_b, 'description')
                _im_emb, _txt_emb = img_out['embedding'].unsqueeze(1), txt_out['embedding'].unsqueeze(1)
                if mm_query == 'addition':
                    _image_emb = _im_emb + _txt_emb
                if mm_query == 'average':
                    _image_emb = (_im_emb + _txt_emb)/2
                im_embed.append(_image_emb)
            else:
                im_embed.append(img_out['embedding'])
        else:
            im_embed.append(img_out['cls_embedding'])
        im_gt += _b['retrieve_gt']

    im_embed = torch.cat(im_embed, dim=0)
    if n_emb > 1 and prob_emb:
        im_logsig = torch.cat(im_logsig, dim=0)

    sent_ret_matrix = None
    ret_matrix = torch.zeros(n_im, n_txt)
    if _config['eval_method'] == 'matmul':
        for idx, query in enumerate(im_embed):
            query = query.unsqueeze(0)
            query = query.view(len(query) * n_emb, -1)
            gallery = txt_embed.view(n_im * n_emb, -1).t()
            _score = query.mm(gallery)
            if n_emb > 1:
                _score = _score.view(
                    len(query)//n_emb, n_emb, gallery.size()[-1]//n_emb, n_emb
                )
                _score = _score.permute(0, 1, 3, 2)
                _score = torch.sum(torch.sum(_score, axis=1), axis=1)
            ret_matrix[idx] = _score
    elif _config['eval_method'] == 'matching_prob':
        scale, shift = pl_module.model.scale, pl_module.model.shift
        for idx, query in enumerate(im_embed):
            _score = match_prob(query.unsqueeze(0), txt_embed, scale, shift)
            ret_matrix[idx] = _score
    elif 'mse_space' in _config['eval_method']:
        s = int(_config['eval_method'].split('_')[-1])
        assert s < n_emb, f"Invalid evaluation method: {_config['eval_method']}."
        ret_matrix = im_embed[:, s, :] @ txt_embed[:, s, :].t()
    elif 'match_sentence' in _config['eval_method']:
        sent_ret_matrix = ret_matrix
        agg = em.split('_')[2]
        if agg  == 'max':
            other = torch.tensor(-1e9).cuda()
            ret_matrix = torch.zeros(n_im, n_im)
            mask_matrix = torch.zeros(n_im, n_txt).bool().cuda()
            for _i in range(n_im):
                mask_matrix[_i] = torch.tensor(np.array(txt_gt) == _i) 
            for idx, query in tqdm.tqdm(enumerate(im_embed), desc="compute score"):
                if 'clip' in _config['losses']:
                    if idx == 0:
                        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
                    query = query / query.norm(dim=-1, keepdim=True)
                    _score = query.unsqueeze(0).mm(txt_embed.t()) * pl_module.model.image_encoder.logit_scale.exp()
                else:
                    query = query.unsqueeze(0)
                    query = query.view(len(query) * n_emb, -1)
                    gallery = txt_embed.view(n_txt * n_emb, -1).t()
                    _score = query.mm(gallery)
                if n_emb > 1:
                    _score = _score.view(
                        len(query)//n_emb, n_emb, gallery.size()[-1]//n_emb, n_emb
                    )
                    _score = _score.permute(0, 1, 3, 2)
                    _score = torch.sum(torch.sum(_score, axis=1), axis=1) # (n_txt)
                sent_ret_matrix[idx] = _score
                _score = _score.expand(n_im, -1)  # (n_im, n_txt)
                ret_matrix[idx] = torch.max(torch.where(mask_matrix, _score, other), dim=-1).values
        elif agg == 'first':
            if 'clip' in _config['losses']:
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                im_embed = im_embed / im_embed.norm(dim=-1, keepdim=True)
                ret_matrix = im_embed.mm(txt_embed.t()) * pl_module.model.image_encoder.logit_scale.exp()
            else:
                ret_matrix = im_embed.mm(txt_embed.t())
        else:
            raise ValueError("Unsupported aggregator.")
        im_gt, txt_gt, n_txt = None, None, n_im
    elif 'clip' == _config['eval_method']:
        ret_matrix = im_embed @ txt_embed.t() * pl_module.model.image_encoder.logit_scale.exp()
    else:
        raise f"Method {_config['eval_method']} is not supported for evaluation."
    # import pudb; pu.db
    ret_matrix = ret_matrix.detach().cpu().numpy()
    i2t = rank(ret_matrix, n_im, refs=im_gt)
    t2i = rank(np.transpose(ret_matrix), n_txt, refs=txt_gt)
    print("i2t retrieval: ", i2t)
    print("t2i retrieval: ", t2i)

    expt_path = os.path.join(
            _config['result_dir'], 'inference', _config['expt_name'], 'retrieve'
        )
    os.makedirs(expt_path, exist_ok=True)
    print(f"Saving retrieval output to: {expt_path}")
    if n_emb > 1:
        if prob_emb:
            if mm_query is None:
                np.save(os.path.join(expt_path, f"image_to_{target}_score_{test_split}"), ret_matrix)
                if sent_ret_matrix is not None:
                    np.save(os.path.join(expt_path, f"image_to_{target}_sentence_score_{test_split}"), sent_ret_matrix)
            else:
                np.save(os.path.join(expt_path, f"image_{source[1]}_to_{target}_score_{test_split}"), ret_matrix)
        else:
            np.save(os.path.join(expt_path, f"image_to_{target}_score_{test_split}"), ret_matrix)
            if sent_ret_matrix is not None:
                np.save(os.path.join(expt_path, f"image_to_{target}_sentence_score_{test_split}"), sent_ret_matrix)
    else: # DE
        if mm_query is None:
            np.save(os.path.join(expt_path, f"image_to_{target}_score_{test_split}"), ret_matrix)
            if sent_ret_matrix is not None:
                np.save(os.path.join(expt_path, f"image_to_{target}_sentence_score_{test_split}"), sent_ret_matrix)
        else:
            np.save(os.path.join(expt_path, f"image_{source[1]}_to_{target}_score_{test_split}"), ret_matrix)
