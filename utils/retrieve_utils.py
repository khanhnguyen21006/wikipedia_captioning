import os
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

@torch.no_grad()
def retrieve_wrapup(pl_module):
    """
	1. call dataloaders
	2. loop over valset -> extract features per batch
	3. gather distributedly-computed features
	4. compute score
    """
    ws = torch.distributed.get_world_size()
    _config = pl_module.hparams._config
    test_split = _config['retrieval_testset']
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
        model_out = pl_module.model.encode(_b)
        if n_emb > 1:
            if prob_emb:
                update_prob_embed(emb_out, model_out, mm_query, source, target, n_emb)
            else:
                update_set_embed(emb_out, model_out, source, target)
        else:
            update_det_embed(emb_out, model_out, mm_query, source, target)
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
        ret_matrix, _ = ret_matrix.max(dim=1)
    elif _config['eval_method'] == 'smooth_chamfer':
        assert n_emb > 1
        alpha = pl_module.model._config['chamfer_alpha']
        image_emb = F.normalize(buffer['image_emb'], dim=-1)
        text_emb = F.normalize(buffer['text_emb'], dim=-1)
        ret_matrix = smooth_chamfer_loss(image_emb, text_emb, alpha, return_score=True)
    else:
        raise f"Method {_config['eval_method']} is not supported for evaluation."

    ret_matrix = ret_matrix.detach().cpu().numpy()
    i2t = rank(ret_matrix, n_query)
    t2i = rank(np.transpose(ret_matrix), n_query)
    print("i2t retrieval: ", i2t)
    print("t2i retrieval: ", t2i)

    if torch.distributed.get_rank() == 0:
        if not n_emb > 1 and not prob_emb:
            return
        expt_path = os.path.join(
            _config['result_dir'], 'inference', _config['expt_name'], 'retrieve'
        )
        os.makedirs(expt_path, exist_ok=True)
        print(f"Saving retrieval output to: {expt_path}")
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

def update_det_embed(emb_out, model_out, mm_query, source, target):
    _image_emb, _text_emb = model_out[source[0]]['embedding'].unsqueeze(1), model_out[target]['embedding'].unsqueeze(1)
    if mm_query is not None:
        _im_emb, _txt_emb = model_out['image']['embedding'].unsqueeze(1), model_out[source[1]]['embedding'].unsqueeze(1)
        if mm_query == 'addition':
            _image_emb = _im_emb + _txt_emb
        if mm_query == 'average':
            _image_emb = (_im_emb + _txt_emb)/2
        emb_out['im_emb'].append(_im_emb)
        emb_out['txt_emb'].append(_txt_emb)
    emb_out['image_emb'].append(_image_emb)
    emb_out['text_emb'].append(_text_emb)
