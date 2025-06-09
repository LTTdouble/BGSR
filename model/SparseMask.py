
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import SparseMask_common


class SparseAttention(nn.Module):
    def __init__(self, n_hashes, channels, chunk_size, res_scale, k_size=3, reduction=4,
                 conv=SparseMask_common.default_conv):
        super(SparseAttention, self).__init__()
        self.chunk_size = chunk_size
        self.n_hashes = n_hashes
        self.reduction = reduction
        self.res_scale = res_scale
        self.conv_match = SparseMask_common.BasicBlock(conv, channels, channels // reduction, k_size, bn=False, act=None)
        self.conv_assembly = SparseMask_common.BasicBlock(conv, channels, channels, 1, bn=False, act=None)

    def LSH(self, hash_buckets, x):
        N = x.shape[0]
        device = x.device

        rotations_shape = (1, x.shape[-1], self.n_hashes, hash_buckets // 2)
        random_rotations = torch.randn(rotations_shape, dtype=x.dtype, device=device).expand(N, -1, -1, -1)

        rotated_vecs = torch.einsum('btf,bfhi->bhti', x, random_rotations)
        rotated_vecs = torch.cat([rotated_vecs, -rotated_vecs], dim=-1)

        hash_codes = torch.argmax(rotated_vecs, dim=-1)

        offsets = torch.arange(self.n_hashes, device=device)
        offsets = torch.reshape(offsets * hash_buckets, (1, -1, 1))
        hash_codes = torch.reshape(hash_codes + offsets, (N, -1,))

        return hash_codes

    def add_adjacent_buckets(self, x):
        x_extra_back = torch.cat([x[:, :, -1:, ...], x[:, :, :-1, ...]], dim=2)
        x_extra_forward = torch.cat([x[:, :, 1:, ...], x[:, :, :1, ...]], dim=2)
        return torch.cat([x, x_extra_back, x_extra_forward], dim=3)

    def forward(self, input):
        N, _, H, W = input.shape
        x_embed = self.conv_match(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        y_embed = self.conv_assembly(input).view(N, -1, H * W).contiguous().permute(0, 2, 1)
        L, C = x_embed.shape[-2:]

        hash_buckets = min(L // self.chunk_size + (L // self.chunk_size) % 2, 128)
        hash_codes = self.LSH(hash_buckets, x_embed)  # [N, n_hashes * H * W]
        hash_codes = hash_codes.detach()

        _, indices = hash_codes.sort(dim=-1)  # [N, n_hashes * H * W]
        _, undo_sort = indices.sort(dim=-1)
        mod_indices = (indices % L)  # Now range from (0->H*W)
        x_embed_sorted = SparseMask_common.batched_index_select(x_embed, mod_indices)  # [N, n_hashes * H * W, C]
        y_embed_sorted = SparseMask_common.batched_index_select(y_embed, mod_indices)  # [N, n_hashes * H * W, C]

        padding = self.chunk_size - L % self.chunk_size if L % self.chunk_size != 0 else 0
        x_att_buckets = torch.reshape(x_embed_sorted, (N, self.n_hashes, -1, C))  # [N, n_hashes, H*W, C]
        y_att_buckets = torch.reshape(y_embed_sorted, (N, self.n_hashes, -1, C * self.reduction))

        if padding:
            pad_x = x_att_buckets[:, :, -padding:, :].clone()
            pad_y = y_att_buckets[:, :, -padding:, :].clone()
            x_att_buckets = torch.cat([x_att_buckets, pad_x], dim=2)
            y_att_buckets = torch.cat([y_att_buckets, pad_y], dim=2)

        x_att_buckets = torch.reshape(x_att_buckets, (
        N, self.n_hashes, -1, self.chunk_size, C))  # [N, n_hashes, num_chunks, chunk_size, C]
        y_att_buckets = torch.reshape(y_att_buckets, (N, self.n_hashes, -1, self.chunk_size, C * self.reduction))

        x_match = F.normalize(x_att_buckets, p=2, dim=-1, eps=5e-5)
        x_match = self.add_adjacent_buckets(x_match)
        y_att_buckets = self.add_adjacent_buckets(y_att_buckets)

        raw_score = torch.einsum('bhkie,bhkje->bhkij', x_att_buckets,
                                 x_match)  # [N, n_hashes, num_chunks, chunk_size, chunk_size]

        # Softmax
        bucket_score = torch.logsumexp(raw_score, dim=-1, keepdim=True)
        score = torch.exp(raw_score - bucket_score)  # (after softmax)
        bucket_score = torch.reshape(bucket_score, [N, self.n_hashes, -1])

        # Select top-k scores
        k = int(self.chunk_size * 0)  # Example: take top 20%
        top_k_scores, top_k_indices = torch.topk(score, k, dim=-1)  # [N, n_hashes, num_chunks, k]

        # Create a tensor to hold the filled scores
        filled_scores = torch.zeros_like(score)  # Initialize filled scores tensor

        # Scatter the top-k scores back into the filled tensor
        filled_scores.scatter_(dim=-1, index=top_k_indices, src=top_k_scores)  # Fill with top-k scores

        # Attention
        ret = torch.einsum('bukij,bukje->bukie', filled_scores,
                           y_att_buckets)  # [N, n_hashes, num_chunks, chunk_size, C]
        ret = torch.reshape(ret, (N, self.n_hashes, -1, C * self.reduction))

        if padding:
            ret = ret[:, :, :-padding, :].clone()
            bucket_score = bucket_score[:, :, :-padding].clone()

        ret = torch.reshape(ret, (N, -1, C * self.reduction))  # [N, n_hashes * H * W, C]
        bucket_score = torch.reshape(bucket_score, (N, -1,))  # [N, n_hashes * H * W]
        ret = SparseMask_common.batched_index_select(ret, undo_sort)  # [N, n_hashes * H * W, C]
        bucket_score = bucket_score.gather(1, undo_sort)  # [N, n_hashes * H * W]

        # Weighted sum multi-round attention
        ret = torch.reshape(ret, (N, self.n_hashes, L, C * self.reduction))  # [N, n_hashes * H * W, C]
        bucket_score = torch.reshape(bucket_score, (N, self.n_hashes, L, 1))
        probs = nn.functional.softmax(bucket_score, dim=1)
        ret = torch.sum(ret * probs, dim=1)

        ret = ret.permute(0, 2, 1).view(N, -1, H, W).contiguous() * self.res_scale + input
        return ret


