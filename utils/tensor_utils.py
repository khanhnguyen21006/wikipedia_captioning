import torch

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

    # Create a ones tensor
    pad_ones = torch.ones(y.size(0), y.size(1) - 1)

    update_index = torch.cat([tmp_idx.unsqueeze(-1), pad_ones.long()], dim=-1)
    return torch.cumsum(update_index, dim=1)