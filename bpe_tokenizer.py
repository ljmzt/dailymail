import numpy as np
import collections
import torchtext.transforms as T
from utils import preprocess_string, SpacyPunctuation
import torch
import pickle
from trie import Trie
from itertools import cycle
import re

class BPETokenizer():
    def __init__(self, filename, vocab_filename, max_tokens, drop=0.0, max_len=3500):
        # set up the specials
        n_specials = 4
        self.unk = '<unk>'
        self.bos = '<bos>'
        self.eos = '<eos>'
        self.pad = '<pad>'
        self.unk_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.pad_idx = 3
        self.totensor = T.ToTensor(padding_value=self.pad_idx, dtype=torch.long)
        self.drops = cycle(np.array(torch.rand(10000) >= drop))
        self.max_len = max_len
        self.split_pat = re.compile('|'.join([self.unk, self.bos, self.eos, self.pad]) + r'|\w+')

        # now load the vocab and set up the trie and the helper sp
        with open(vocab_filename, 'rb') as fid:
            tmp = pickle.load(fid)
        self.unique_chars = tmp['uniq_chars']
        vocab = tmp['vocab']
        self.vocab = sorted(vocab, key=lambda x:(vocab[x], len(x), x))
        i = 0
        for _ in range(n_specials):
            while self.vocab[i] in self.unique_chars:  i += 1
            self.vocab.pop(i)
        self.vocab = [self.unk, self.bos, self.eos, self.pad] + self.vocab[::-1]      
        self.vocab = self.vocab[:max_tokens]
        self.sp = SpacyPunctuation(tmp['space'], tmp['punctuations'])

        # construct the trie
        self.trie = Trie()
        self.trie.construct([word[::-1] for word in self.vocab], 
                             list(range(max_tokens))) 
        self.dp = [0 for _ in range(20000)]
        self.dp_idx = [-1 for _ in range(20000)]

        # now loads and pre_process the lines
        if filename is None:
            return
        self.stories, self.summaries = [], []
        with open(filename, 'r') as fid:
            for i, line in enumerate(fid):
                if i == 0:  continue
                if i % 1000 == 0:  print(i)
                try:
                    story, summary = line.split('\t')
                except:
                    continue
                if len(story.split()) <= len(summary.split()):
                    print("----- BAD STUFF -----")
                    print(story, summary)
                    continue
                story = preprocess_string(story.strip())
                summary = preprocess_string(summary.strip())
                self.stories.append(self.sp(story))
                self.summaries.append(self.sp(summary))

    def tokenize(self, string, preprocessed=True):
        if not preprocessed:
            string = preprocess_string(string.strip())
            string = self.sp(string)
            print('preprocess',string)
        n = len(string)
        self.dp[0] = 0
        for j in range(1, n+1):
            i = j - 1
            node = self.trie
            dp_min = 999999
            while i >= 0:
                c = string[i]
                if c not in node.children:  # stop if it can't go further back
                    break
                else:
                    node = node.children[c]
                    # update dp only when this is a valid word and not randomly dropped
                    # single character is always kept to avoid unk
                    if i == j - 1 or (node.val is not None and next(self.drops)):
                        if self.dp[i] < dp_min:
                            dp_min = self.dp[i]
                            idx_sv = node.val
                    i -= 1                
            self.dp[j] = dp_min + 1
            self.dp_idx[j] = idx_sv
        # get the pieces
        i = n
        output = []
        while i > 0:
            idx = self.dp_idx[i]
            output.append(idx)
            i -= len(self.vocab[idx])
        return output[::-1][:self.max_len]

    def detokenize(self, x):
        ''' x is a list of integer '''
        return ''.join([self.vocab[i] for i in x])

    def detokenize_as_piece(self, x):
        ''' x is a list of integer '''
        return '@@'.join([self.vocab[i] for i in x])

    def detokenize_as_split(self, x):
        ''' x is a list of integer '''
        string = self.detokenize(x)
        return self.split_pat.findall(string)

if (__name__ == '__main__'):
    in_file = 'dataset/dailymail/onebigfile_validation.txt'
    out_file = 'dataset/dailymail/onebigfile_validation.npy'
    n_rounds = 1
    tokenizer = BPETokenizer(in_file,
                             'BPE_vocab_10000.pickle',
                             max_tokens = 10000,
                             drop = 0.0,
                             max_len = 5000)

    output = []
    for i, (story, summary) in enumerate(zip(tokenizer.stories, tokenizer.summaries)):
        if i % 1000 == 0:
            print(i)
        for _ in range(n_rounds):
            arr1 = np.array(tokenizer.tokenize(story), dtype=np.uint16)
            arr2 = np.array(tokenizer.tokenize(summary), dtype=np.uint16)
            output.append(arr1)
            output.append(arr2)

    np.save(out_file, np.array(output, dtype=object), allow_pickle=True)


    in_file = 'dataset/dailymail/onebigfile_train.txt'
    out_file = 'dataset/dailymail/onebigfile_train.npy'
    n_rounds = 10
    tokenizer = BPETokenizer(in_file,
                             'BPE_vocab_10000.pickle',
                             max_tokens = 10000,
                             drop = 0.1,
                             max_len = 5000)

    output = []
    for i, (story, summary) in enumerate(zip(tokenizer.stories, tokenizer.summaries)):
        if i % 1000 == 0:
            print(i, datetime.now())
        for _ in range(n_rounds):
            arr1 = np.array(tokenizer.tokenize(story), dtype=np.uint16)
            arr2 = np.array(tokenizer.tokenize(summary), dtype=np.uint16)
            output.append(arr1)
            output.append(arr2)

    np.save(out_file, np.array(output, dtype=object), allow_pickle=True)
