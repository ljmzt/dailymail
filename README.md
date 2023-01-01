# CNN Daily Mail -- Text Summarization
## Purpose
Code up the transformer architecture for text summary. Can't believe it has taken me nearly 4 months.

## Data
Use the widely used CNN daily mail dataset, downloaded from hugging face.

Preprocess script -- parse_cnn.ipynb, combine files into onebigfile.txt.

The file is pretty big and exceed github limit. I put a copy of it on the kaggle dataset.

## Tokenizer 
Well, the first couple months are actually used to construct a BPE tokenizer, sigh.

Some helpful resources are [Lei Mao's blog](https://leimao.github.io/blog/Byte-Pair-Encoding/), [Harshit Tyagi's](https://www.freecodecamp.org/news/evolution-of-tokenization/), and the source code from [Rico Sennrich](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py).

Most implementations suggest splitting into words (by space) first, which I don't think quite necessary. The new part in my implementation is to use 1) dynamic programming in the encoding part to generate a tokenization scheme that results in minimal numbers of tokens, and 2) a trie to keep track of how far backward one needs to go. This avoid the many str.index calls, which are extremely time consuming.

To further facilitate training, I also precompute the tokens, save them into npy files, so don't need to compute these again and again.

[BPE-dropout](https://arxiv.org/abs/1910.13267) is already included.

Main codes:
bpe_construct.py: construct the vocab, similar to existing implementations. 
bpe_tokenizer.py: implement dynamic programming + trie for much faster encoding. It also precompute the tokenized files used later for training/validation. 
bpe-create-vocab.ipynb: the notebook that calls bpe_construct.py to build up the vocab.

## Various Transformer
The next 
