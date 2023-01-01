# construct the vocabulary for tokenizer
import numpy as np
import collections
from utils import SpacyPunctuation, preprocess_string
import pickle

class ISentence():
    def __init__(self, sentence):
        self.string = sentence
        self.n = len(sentence)
        
        # marks the end of this byte piece so idx:end[idx] is one piece
        self.end = np.array([i+1 for i in range(self.n)], dtype=np.uint16)
        self.start = np.array([i for i in range(self.n)], dtype=np.uint16)
        
        # indexes[pair] is the position this pair appear, can have multiple positions
        self.indexes = collections.defaultdict(set)
        for i in range(self.n-1):
            j = i + 1
            pair = (sentence[i], sentence[j])
            self.indexes[pair].add(i)
    
    def get_curr(self, idx):
        if idx >= self.n or idx < 0:
            return -1, 'NAN'
        else:
            return idx, self.string[idx:self.end[idx]]
    
    def get_next(self, idx):
        idx = self.end[idx]
        if idx >= self.n:
            return -1, 'NAN'
        else:
            return self.get_curr(idx)
    
    def get_prev(self, idx):
        if idx == 0:
            return -1, 'NAN'
        else:
            idx = self.start[idx-1]
            return self.get_curr(idx)
    
    def merge(self, idx):
        '''
           merge the idx piece and the next piece
        '''
        j1 = self.end[idx]
        if j1 == self.n:
            return
        j2 = self.end[j1]
        for i in range(idx, j1):
            self.end[i] = j2        
        for j in range(j1, j2):
            self.start[j] = idx
    
    def merge_pair(self, pair):
        '''
           merge the given pair and adjus all the indexes
           the difficult part is to make sure 
           1) A B C B C -> A BC BC
           2) A A A -> AA A
           3) A B C D B C -> A BC D BC
        '''
        first, second = pair
        todo = []
        for i in sorted(list(self.indexes[pair])):
            iprev, _ = self.get_prev(i)
            if todo and todo[-1] == iprev:   # deal with A A A 
                continue
            todo.append(i)
            
        # deal with A B C B C cases here
        specials = ['' for _ in range(len(todo))]
        for i in range(len(todo)-1):
            next2 = self.get_next(self.get_next(todo[i])[0])[0]
            #print('checking', next2, todo[i+1])
            if next2 == todo[i+1]:
                specials[i] += 'F'   # don't look forward
                specials[i+1] += 'B'  # merge stuff when look backward
                
        changes = collections.defaultdict(int)
        
        for i, special in zip(todo, specials):
            #print(f'doing position:{i},{special}')
            iprev, wprev = self.get_prev(i)
            
            # look backward
            if iprev >= 0:
                # remove
                old_pair = (wprev, first)
                #print('remove back:',pair, old_pair, iprev)
                self.indexes[old_pair].remove(iprev)
                changes[old_pair] -= 1
                if len(self.indexes[old_pair]) == 0:
                    self.indexes.pop(old_pair)
                
                # delayed merging
                if 'B' in special:
                    iprev2, _ = self.get_prev(iprev)
                    self.merge(iprev2)
                    wprev = first + second
                    iprev = iprev2
                
                # add
                new_pair = (wprev, first + second)
                #print('add back:', pair, new_pair, iprev)
                self.indexes[new_pair].add(iprev)
                changes[new_pair] += 1
                
            # look forward
            j, _ = self.get_next(i)
            jnext, wnext = self.get_next(j)
            if jnext > 0:                
                # don't look forwad
                if 'F' in special:
                    continue
                
                # remove
                old_pair = (second, wnext)
                #print('remove forward:',pair, old_pair, j)
                self.indexes[old_pair].remove(j)
                changes[old_pair] -= 1
                if len(self.indexes[old_pair]) == 0:
                    self.indexes.pop(old_pair)
                
                # add
                new_pair = (first + second, wnext)
                #print('add forward:', pair, new_pair, i)
                self.indexes[new_pair].add(i)
                changes[new_pair] += 1
                
            # merge
            self.merge(i)
        
        # lastly remove this pair from the indexes
        changes[pair] -= len(self.indexes[pair])
        self.indexes.pop(pair)
        return changes
            
    def show(self):
        i = 0
        output = []
        while i < self.n:
            output.append(self.string[i:self.end[i]])
            i = self.end[i]
        return output

def construct(filename, space, punctuations,
              first_lines = -1,
              vocab_sizes = [10000,15000,20000],
              allow_cross = False,
              prefix = 'BPE_vocab'):
    unique_chars = set()
    sentences = []
    sp = SpacyPunctuation(space, punctuations)
    
    # reads in the lines
    with open(filename, 'r') as fid:
        for i, line in enumerate(fid):
            if i == 0:  continue
            try:
                line = line.split('\t')[0].strip()
            except:
                continue
            line = preprocess_string(line)
            line = sp(line)
            unique_chars |= set(line)
            sentences.append(ISentence(line))
            if first_lines > 0 and i > first_lines:
                break
    print('done reading in sentences')
    
    # init the counts for pair and vocab
    counts = collections.defaultdict(int)
    vocab = collections.defaultdict(int)
    for sentence in sentences:
        for c in sentence.string:
            vocab[c] += 1
        for pair, indexes in sentence.indexes.items():
            counts[pair] += len(indexes)
    print('done init counts')
    
    # now do whole bunches of merges
    nmerge = 0
    while counts:
        nmerge += 1
        pair = max(counts, key=counts.get)
        print(f"{nmerge} merging: ->{pair}<- {counts[pair]}")
        first, second = pair
        if first in punctuations or second in punctuations:
            print(f"skip")
            counts.pop(pair)
            continue
        if allow_cross == False and first[-1] == space:
            print(f"skip")
            counts.pop(pair)
            continue
        
        # merge and fix the counts
        nums = 0
        for sentence in sentences:
            changes = sentence.merge_pair(pair)
            nums -= changes[pair]
            for k, v in changes.items():
                counts[k] += v
        
        # update the vocab
        vocab[first] -= nums
        vocab[second] -= nums
        vocab[first+second] += nums
        if vocab[first] == 0:  vocab.pop(first)
        if vocab[second] == 0:  vocab.pop(second)
        
        if len(vocab) >= vocab_sizes[0]:
            outfile = prefix + '_' + str(vocab_sizes[0]) + '.pickle'
            with open(outfile, 'wb') as fid:
                pickle.dump({'uniq_chars':unique_chars,
                             'vocab': vocab,
                             'space': space,
                             'punctuations': punctuations}, fid)
            vocab_sizes.pop(0)
            if len(vocab_sizes) == 0:  break
