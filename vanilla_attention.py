'''
vanilla attention 
'''
import torch.nn as nn
import numpy as np
import torch

class Attention(nn.Module):
    def __init__(self, dmodel, dk, dhead, share_weight):
        ''' 
          dmodel = embedding size
          dk = dimension for each head
          dhead = number of head
          dk*dhead does not necessarily equal to dmodel
          if share_weight, WQ=WK
        '''
        super().__init__()
        self.dk = dk
        self.dhead = dhead
        self.depth = self.dk * self.dhead
        self.WQ = nn.Linear(dmodel, self.depth)
        if share_weight:
            self.WK = self.WQ
        else:
            self.WK = nn.Linear(dmodel, self.depth)
        self.WV = nn.Linear(dmodel, self.depth)
        self.WO = nn.Linear(self.depth, dmodel)
        self.fac = 1.0/np.sqrt(self.dk)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, y, mask=None):
        '''
          x is the query
          x = N * Lq * dmodel
          y = N * L * dmodel containing the keys
          x and y can be the same
          output size will be N * Lq * dmodel for x, and N * dhead * Lq * L for attn mask
          the mask controls the causality
        '''
        q = self.WQ(x)
        k = self.WK(y)
        v = self.WV(y)

        N, Lq, _ = x.shape
        _, L, _ = y.shape
        q = q.reshape(N, Lq, self.dhead, self.dk).transpose(1, 2).contiguous()
        k = k.reshape(N, L, self.dhead, self.dk).transpose(1 ,2).contiguous().transpose(2, 3)
        v = v.reshape(N, L, self.dhead, self.dk).transpose(1, 2).contiguous()
        x = q @ k * self.fac   # N * dhead * Lq * L

        if mask is not None:
            x = x.transpose(0, 1).masked_fill(mask==False, -1e+9).transpose(0, 1)
        x = self.softmax(x)

        x = x @ v  # N * dhead * Lq * dk
        x = x.transpose(1, 2).reshape(N, Lq, self.depth)
        x = self.WO(x)

        return x
