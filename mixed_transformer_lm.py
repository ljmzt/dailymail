'''
The mixed tranformer language model
use fft for the encoder, performer for the crossdecoder, vanilla for the selfdecoder
dmodel is the embedding size
max_len is for positional encoding
'''
import torch.nn as nn
import numpy as np
import torch
from datetime import datetime
from utils import get_time

from vanilla_attention import Attention
from fft_attention import F_Attention
from performer_sharecontext_attention import P_Attention

class PositionalEncoding(nn.Module):
    def __init__(self, max_len, dmodel):
        super().__init__()
        pos = torch.arange(0, max_len)[:,None]
        power = torch.exp(torch.arange(0, dmodel, 2) * (-np.log(10000.0) / dmodel))
        tmp = pos * power
        pe = torch.zeros(max_len, dmodel)
        pe[:, 0::2] = torch.sin(tmp)
        pe[:, 1::2] = torch.cos(tmp)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        '''
          x = N * L * dmodel
        '''
        return self.pe[:x.shape[1]]

class FeedForward(nn.Module):
    def __init__(self, dmodel, dff):
        super().__init__()
        self.fc1 = nn.Linear(dmodel, dff)
        self.fc2 = nn.Linear(dff, dmodel)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        '''
        x = N * L * dmodel
        '''
        x = self.activation(self.fc1(x))
        return self.fc2(x)

class Encoder(nn.Module):
    def __init__(self, dmodel, dff, p=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.attention = F_Attention()  # no learned parameters
        self.ff = FeedForward(dmodel, dff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)

    def forward(self, x):
        z = self.attention(x)
        z = self.norm1(z + x)
        z = self.dropout(z)
        output = self.ff(z)
        output = self.norm2(output + z)
        return self.dropout(output)

class Decoder(nn.Module):
    def __init__(self, dmodel, dk, dhead, dff, p=0.1, kernel_type='relu', share_context=False):
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.cross_attention = P_Attention(dmodel, dk, dhead,
                                     kernel_type=kernel_type, share_context=share_context)
        self.self_attention = Attention(dmodel, dk, dhead, share_weight=True)
        self.ff = FeedForward(dmodel, dff)
        self.norm1 = nn.LayerNorm(dmodel)
        self.norm2 = nn.LayerNorm(dmodel)
        self.norm3 = nn.LayerNorm(dmodel)

    def forward(self, x, y):
        '''
           x is the query 
           y is the encoder output, or (k,v) pair when share_context
        '''

        L = x.shape[1]
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)
        z = self.self_attention(x, x, mask)
        z = self.dropout(self.norm1(z + x))

        zz, (k, v), attn = self.cross_attention(z, y)
        zz = self.dropout(self.norm2(zz + z))

        output = self.ff(zz)
        output = self.dropout(self.norm3(output + zz))
        return output, (k, v), attn 

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, max_len1, max_len2, dmodel, dk, dhead, dff, p, nlayer, kernel_type): 
        super().__init__()
        self.encoder_layers = nn.ModuleList([Encoder(dmodel, dff, p) for _ in range(nlayer)])
        self.decoder_layers = []
        self.decoder_layers.append(Decoder(dmodel, dk, dhead, dff, 
                                   kernel_type=kernel_type,share_context=False))
        for _ in range(1, nlayer):
            self.decoder_layers.append(Decoder(dmodel, dk, dhead, dff, 
                                       kernel_type=kernel_type, share_context=True))
        self.decoder_layers = nn.ModuleList(self.decoder_layers)
        self.dropout = nn.Dropout(p=p)
        self.embedding = nn.Embedding(vocab_size, dmodel)
        self.pe1 = PositionalEncoding(max_len1, dmodel)
        self.pe2 = PositionalEncoding(max_len2, dmodel)

        # assume weight sharing here
        #self.W = self.embedding.weight.T/np.sqrt(dmodel)   # dmodel, vocab_size
        self.linear = nn.Linear(dmodel, vocab_size, bias=False)
        self.linear.weight = self.embedding.weight
        self.linear_fac = 1.0 / np.sqrt(dmodel)
        
    def forward(self, x, y):
        '''
           x is the story
           y is the summary
           the return is logit
        ''' 
        #t1 = datetime.now()
        x = self.embedding(x)
        x = self.dropout(x + self.pe1(x))
        for layer in self.encoder_layers:
            x = layer(x)
        encoder_output = x
        #t2 = datetime.now()
        #print(f'encoder total {get_time(t1,t2)}')

        #t1 = datetime.now()
        y = self.embedding(y)
        y = self.dropout(y + self.pe2(y))
        y, (k,v), attn = self.decoder_layers[0](y, encoder_output)
        #t2 = datetime.now()
        #print(f'decoder first {get_time(t1,t2)}')

        #t1 = datetime.now()
        for layer in self.decoder_layers[1:]:
            y, _, attn = layer(y, (k,v))
        decoder_output = y
        #t2 = datetime.now()
        #print(f'decoder rest {get_time(t1,t2)}')
    
        #output = x @ self.W
        output = self.linear(decoder_output) * self.linear_fac  # N * L * vocab_size
        return output, attn   # return the attn for the last layer
