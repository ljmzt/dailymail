import torch
from torchtext.data.metrics import bleu_score
import re
import unicodedata
import numpy as np
from datetime import datetime

class SpacyPunctuation():
    ''' 
       we normally allow word+\s, but not \.\s, 
       so need to remove the space after the punctuation and add to the front
    '''
    def __init__(self, space, punctuations):
        self.space = space
        if self.space != '':
            self.subpat = '|'.join([re.escape(p) for p in punctuations])
            # add a space before the punctuation
            self.subpat1 = r'(?<=\S)' + '(' + self.subpat + ')'
            self.subpat1 = re.compile(self.subpat1)
            self.repl1 = space + '\\1'
            # remove the space after
            self.subpat2 = '(' + self.subpat + ')' + r'\s'
            self.subpat2 = re.compile(self.subpat2)
            self.repl2 = '\\1'

    def __call__(self, string):
        if self.space == '':
            return string
        else:
            string = self.subpat1.sub(self.repl1, string)
            string = self.subpat2.sub(self.repl2, string)
            return string 

def preprocess_string(string, lower=True, toascii=True, fullstop=True):
    # remove white space
    string = re.sub('\s+',' ',string)
    # convert to ascii
    if toascii:
        string = unicodedata.normalize('NFKD', string).encode('ascii','ignore').decode()
    # lower
    if lower:
        string = string.lower()
    return string
    

def get_coverage(attn):
    '''
       attn = N * nhead * Lq * L
       return the coverage score, reduction is mean
    '''
    attn_cumsum = torch.cumsum(attn, dim=2)
    return torch.mean(torch.minimum(attn[:,:,1:,:].reshape(-1),
                                    attn_cumsum[:,:,:-1,:].reshape(-1)))

def get_bleu_score(tokenizer, output, z):
    '''
       output = N * L, which is after torch.argmax
       z = N * L
       only non-pad words are counted
       device is in cpu
       according to the doc, output, z, w should contain the entire corpus, quite strange
    '''
    N = len(output)
    candidates, targets = [], []
    for i in range(N):
        candidates.append(tokenizer.detokenize_as_split(output[i]))
        targets.append([tokenizer.detokenize_as_split(z[i])])
    return bleu_score(candidates, targets)

def get_size(model):
    ''' assume torch.float32, 4 bytes '''
    model_size = 0
    for name, p in model.named_parameters():
        print(name, p.shape, p.dtype)
        model_size += np.product(p.shape)*4/1024/1024
    print(f"total size {model_size}Mb")

def get_time(t1, t2):
    return (t2-t1).seconds + (t2-t1).microseconds/1e+6
