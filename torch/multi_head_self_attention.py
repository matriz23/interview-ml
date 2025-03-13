import numpy as np
import torch
import torch.nn as nn




class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_in, dim_k, dim_v, n_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % n_heads == 0 and dim_v % n_heads == 0
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.n_heads = n_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / np.sqrt(dim_k // n_heads)
    
    def forward(self, x):
        # x: (bs, n, dim_in)
        bs, n, dim_in = x.shape
        assert dim_in == self.dim_in
        
        nh = self.n_heads
        dk = self.dim_k // nh
        dv = self.dim_v // nh
        
        
        q = self.linear_q(x).reshape(bs, n, nh, dk).transpose(1, 2) 
        # q: (bs, nh, n, dk)
        k = self.linear_k(x).reshape(bs, n, nh, dk).transpose(1, 2) 
        v = self.linear_v(x).reshape(bs, n, nh, dv).transpose(1, 2)
        

        dist = torch.matmul(q, k.transpose(2, 3)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        # dist: (bs, nh, n, n)
        
        att = torch.matmul(dist, v)
        # att: (bs, nh, n, dv)
        
        att = att.transpose(1, 2).reshape(bs, n, self.dim_v)
        # att: (bs, nh, dim_v)
        return att