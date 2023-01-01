'''
the FNet attention is simple
there is no learned parameter
and indeed doing fft is faster compared with matrix multiplication on GPU
'''
import torch.nn as nn
import torch

class F_Attention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''
           x = N * L * dmodel
           do a fft 
        '''
        return torch.real(torch.fft.fft2(x, norm='ortho'))
