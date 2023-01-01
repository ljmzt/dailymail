# implement the performer model
# ref
# https://hub.nuaa.cf/lucidrains/performer-pytorch/blob/main/performer_pytorch/performer_pytorch.py
# https://hub.nuaa.cf/nawnoes/pytorch-performer/blob/main/model/performer.py
# to further reduce the cost, can assume the k, v are the same throughout different layers
import torch
import numpy as np
from torch.nn.functional import relu
from torch import nn

def gen_projection(m, d, ortho=True):
    '''
       generate the OPF of d*m, with m<=d
       the return will have a norm of sqrt(d)
       return shape d * m, so the operation later is x @ matrix
    '''
    if ortho == False:
        return torch.randn((d, m))
    q, _ = torch.linalg.qr(torch.randn(d,d))
    matrix = q[:,:m] * np.sqrt(d)
    matrix = matrix.contiguous()
    return matrix

def kernel_transform(x, omega, kernel_type='relu', fac1=1.0, fac2=1.0, eps=1e-6):
    '''
       calculate the phi(x) according to eqn(5)
       x : N * dhead * L * d
       omega is the projection matrix d * m
       fac1 = 1.0/np.sqrt(m) in eqn(5)
       fac2 = 1.0/np.sqrt(np.sqrt(d)) for the softmax kernel
    ''' 
    if kernel_type == 'relu':
        x = relu(x @ omega)   # N * dhead * L * m
        #x = x * fac1  # this is in omega already for relu type
    elif kernel_type == 'softmax':
        x = x * fac2
        diag = torch.sum(torch.square(x),dim=-1,keepdim=True)/2.0
        x = x @ omega
        x = torch.exp(x - diag)
        x = x * fac1
    else:
        print('not implemented')
    return x + eps

class P_Attention(nn.Module):
    def __init__(self, dmodel, dk, dhead, kernel_type='relu', share_context=False):
        '''
           assume share weight=False, as this is implemented for cross decoder
           kernel_type can be vanilla, relu or softmax
           if share_context, the input for the forward would be x, (k, v), instead of x, y
        '''
        self.share_context = share_context
        super().__init__()
        self.dk = dk
        self.dhead = dhead
        self.depth = self.dk * self.dhead
        self.WQ = nn.Linear(dmodel, self.depth)
        if share_context == False:
            self.WK = nn.Linear(dmodel, self.depth)
            self.WV = nn.Linear(dmodel, self.depth)
        self.WO = nn.Linear(self.depth, dmodel)

        self.kernel_type = kernel_type
        self.fac1 = 1.0/np.sqrt(dk)
        self.fac2 = 1.0/np.sqrt(np.sqrt(dk)) 
        if kernel_type == 'vanilla':
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.reset()

    def forward(self, x, y):
        '''
           x is the query
           x = N * Lq * dmodel
           y = N * L * dmodel containing the keys or (k,v) from previous layer when share_context=True
           output size will be N * Lq * dmodel for x, and N * dhead * Lq * L for attn mask
           no masking is applied
        '''
        q = self.WQ(x)
        if self.share_context:
            k, v = y
        else:
            k = self.WK(y)
            v = self.WV(y)

        N, Lq, _ = x.shape
        q = q.reshape(N, Lq, self.dhead, self.dk).transpose(1, 2)
        if self.share_context:
            L = k.shape[2]
        else:
            _, L, _ = y.shape
            k = k.reshape(N, L, self.dhead, self.dk).transpose(1 ,2).contiguous()
            v = v.reshape(N, L, self.dhead, self.dk).transpose(1, 2).contiguous() # N * dhead * L * dk

        if self.kernel_type == 'vanilla': 
            x = q @ k.transpose(2,3) * self.fac1  #N*dhead*Lq*L
            attn = self.softmax(x)
            x = attn @ v
        else:
            q = kernel_transform(q, self.omega, self.kernel_type, self.fac1, self.fac2)  # N * dhead * Lq * dk
            if self.share_context == False:
                k = kernel_transform(k, self.omega, self.kernel_type, self.fac1, self.fac2)  # N * dhead * L * dk
            d_inv = q @ (k.transpose(2, 3) @ torch.ones(L, 1, device=k.device)) # N * dhead * Lq * 1
            x = (q @ (k.transpose(2,3) @ v)) / d_inv
            attn = None

        x = x.transpose(1, 2).reshape(N, Lq, self.depth)
        x = self.WO(x)
        return x, (k, v), attn


    def reset(self):
        omega = gen_projection(self.dk, self.dk)
        if self.kernel_type == 'relu':
            omega = omega * self.fac1
        self.register_buffer('omega', omega) 
