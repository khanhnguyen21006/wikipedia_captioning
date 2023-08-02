import torch
import numpy as np

def batchwise_cosine_distance(x, y, eps=1e-6):
    if len(x.size()) != 3 or len(y.size()) != 3:
        raise RuntimeError('expected: 3-dim tensors, got: {}, {}'.format(x.size(), y.size()))
    if x.size(0) == y.size(0):
        bs = x.size(0)
    elif x.size(0) == 1:
        bs = y.size(0)
    elif y.size(0) == 1:
        bs = x.size(0)
    else:
        raise RuntimeError(f'x ({x.size()}) and y ({y.size()}) dimensionalities are non-broadcastable.')
    x = x.unsqueeze(1)
    y = y.unsqueeze(2)
    return torch.sqrt(((x - y) ** 2).sum(-1) + eps).view(bs, -1)

def pairwise_sampling(anchors, candidates):
    N = len(anchors)
    if len(anchors) != len(candidates):
        raise RuntimeError('# anchors ({}) != # candidates ({})'.format(anchors.shape, candidates.shape))
    anchor_idx, selected_idx, matched = full_sampling(N)

    anchor_idx = torch.from_numpy(np.array(anchor_idx)).long()
    selected_idx = torch.from_numpy(np.array(selected_idx)).long()
    matched = torch.from_numpy(np.array(matched)).float()

    anchor_idx = anchor_idx.to(anchors.device)
    selected_idx = selected_idx.to(anchors.device)
    matched = matched.to(anchors.device)

    anchors = anchors[anchor_idx]
    selected = candidates[selected_idx]

    return anchors, selected, matched

def full_sampling(n):
    candidates = []
    selected = []
    matched = []
    for i in range(n):
        for j in range(n):
            candidates.append(i)
            selected.append(j)
            if i == j:
                matched.append(1)
            else:
                matched.append(-1)
    return candidates, selected, matched

def sample_gaussian_tensors(mu, logsig, n):
    eps = torch.randn(mu.size(0), n, mu.size(1), dtype=mu.dtype, device=mu.device)
    samples = eps.mul(torch.exp(logsig.unsqueeze(1))).add_(mu.unsqueeze(1))
    return samples

def product_2_gaussians(mu1, logsig1, mu2, logsig2):
    if len(mu1.shape) == 1:
        mu1 = mu1.unsqueeze(0)
    if len(logsig1.shape) == 1:
        logsig1 = logsig1.unsqueeze(0)
    if len(mu2.shape) == 1:
        mu2 = mu2.unsqueeze(0)
    if len(logsig2.shape) == 1:
        logsig2 = logsig2.unsqueeze(0)

    sig1 = torch.exp(logsig1), 
    sig2 = torch.exp(logsig2)
    target_mu = mu2
    target_sig = sig2

    inv_sig1 = 1 / sig1
    inv_target_sig = 1 / target_sig
    C = torch.diag_embed(1 / (inv_sig1 + inv_target_sig))
    c = torch.matmul(C, torch.matmul(torch.diag_embed(inv_sig1), mu1[:, :, None]) + 
        torch.matmul(torch.diag_embed(inv_target_sig), target_mu[:, :, None])).squeeze()
    log_Z = MultilogsigiateNormal(target_mu, torch.diag_embed(sig1 + target_sig + 1e-6)).log_prob(mu1)
    C = torch.diagonal(C, dim1=-2, dim2=-1)
    C = torch.log(C)
    return c, C, log_Z

def addition_2_gaussians(mu1, logsig1, mu2, logsig2):
    sig1, sig2 = torch.exp(logsig1), torch.exp(logsig2)
    return mu1 + mu2, torch.log(sig1 + sig2), torch.zeros(mu1.shape[0], device='cuda')

def mixture_2_gaussians(mu1, logsig1, mu2, logsig2):
    sig1, sig2 = torch.exp(logsig1), torch.exp(logsig2)
    mu = (mu1 + mu2)/2
    logsig = torch.log(
        (sig1 + torch.square(mu1) + sig2 + torch.square(mu2))/2 - torch.square(mu)
    )
    return mu, logsig, torch.zeros(mu1.shape[0], device='cuda')

def log_gaussian(x, mu=0, logsig=0.):
    z = -0.5 * np.log(2 * np.pi)
    if type(logsig) == 'float':
        logsig = x.new(1).fill_(logsig)

    a = (x - mu) ** 2
    log_p = -0.5 * (logsig + a / logsig.exp())
    log_p = log_p + z
    return log_p

def log_1d_mixture_gaussian(x, om, mu, logsig, log=True):
    """
        :param x: design matrix (b, m)
        :param om: the weight of each components (b, h, n, K)
        :param mu: the component means (b, h, n, K)
        :param logvar: the component log-variances (b, h, n, K)
        :param log: return value in log domain?
            Note: exponentiating can be unstable in high dimensions.
        :return likelihoods: (b, h, n)
    """
    # feature-wise log-likelihoods

    log_p = log_gaussian(
        x[:, None, None, :, None],  # (b, 1, 1, m, 1)
        mu[:, :, :, None ,:],  # (b, h, n, 1, K)
        logsig[:, :, :, None ,:]  # (b, h, n, 1, K)
    ) # (b, h, n, m, K)

    log_p = (om[:, :, :, None ,:] * log_p).sum(-1) # (b, h, n, m)
    assert ~any(log_p < 0)
    if not log:
        log_p = log_p.exp()
    return log_p

def trace(x):
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace

def merge_padded_tensors(x, y, pad=0):
    x = x.masked_fill(x == pad, 0)
    y = y.masked_fill(y == pad, 0)
    concated_tensor = torch.cat([x, y], dim=-1)

    # create index tensor of y elements in merged
    index = find_index(x, y)

    merged = concated_tensor.scatter_(1, index, y)
    _x_lens = torch.count_nonzero(x, dim=1)
    _y_lens = torch.count_nonzero(y, dim=1)
    merged_len = length_to_mask(_x_lens + _y_lens, max_len=merged.shape[1])
    merged[~merged_len] = pad
    return merged

def length_to_mask(length, max_len=None):
    assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = length.max() if not max_len else max_len
    return torch.arange(max_len)[None, :].to(length.device) < length[:, None]

def find_index(x, y):
    tmp_idx = torch.bincount((x != 0).nonzero()[:, 0], minlength=x.shape[0])
    pad_ones = torch.ones(y.size(0), y.size(1) - 1)
    update_index = torch.cat([tmp_idx.unsqueeze(-1), pad_ones.long()], dim=-1)
    return torch.cumsum(update_index, dim=1)

def gelu(x):
    """
        Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))\

def distributed_all_gather(x):
    """
    Gathers tensors of same lengths in a tensor.
    """
    ws = torch.distributed.get_world_size()
    all_tensors = [torch.zeros_like(x) for _ in range(ws)]
    torch.distributed.all_gather(all_tensors, x)
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

def distributed_all_gather_nd(x):
    """
    Gathers tensors of different lengths in a tensor.
    The length dimension is 0. This supports any number of extra dimensions in the tensors.
    """
    ws = torch.distributed.get_world_size()
    local_size = torch.tensor(x.size(), device=x.device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    torch.distributed.all_gather(all_sizes, local_size)

    max_len = max(_size[0] for _size in all_sizes)

    len_diff = max_len.item() - local_size[0].item()
    if len_diff:
        pad_size = (len_diff, *x.size()[1:])
        pad = torch.zeros(pad_size, device=x.device, dtype=x.dtype)
        x = torch.cat((x, pad))

    all_padded = [torch.zeros_like(x) for _ in range(ws)]
    torch.distributed.all_gather(all_padded, x)
    all_tensors = []
    for _tensor, _size in zip(all_padded, all_sizes):
        all_tensors.append(_tensor[:_size[0]])
    all_tensors = torch.cat(all_tensors, dim=0)
    return all_tensors

def repeat_tensor_batch_dim(n, x):
    """ (b, ...) ==> (bxn, ...)"""
    x = x.unsqueeze(1)
    x = x.expand(-1, n, *([-1]*len(x.shape[2:])))
    x = x.reshape(x.shape[0]*n, *x.shape[2:])
    return x
