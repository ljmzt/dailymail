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

Most implementations suggest first splitting into words (in English, this means splitting using space as a separator), which I don't think is quite necessary. At the end of the day, there're quite a few languages that can't be easily split. The new part in my implementation is to use 1) dynamic programming in the encoding part to generate a tokenization scheme that results in minimal numbers of tokens, and 2) a trie to keep track of how far backward one needs to go. This avoid the many str.index calls, which are extremely time consuming.

To further facilitate training, I also precompute the tokens, save them into npy files, so don't need to compute these again and again. The training/validation files are again on the kaggle dataset.

[BPE-dropout](https://arxiv.org/abs/1910.13267) is already included.

Main codes:
bpe_construct.py: construct the vocab, similar to existing implementations. 
bpe_tokenizer.py: implement dynamic programming + trie for much faster encoding. It also precompute the tokenized files used later for training/validation. 
bpe-create-vocab.ipynb: the notebook that calls bpe_construct.py to build up the vocab.

## Various Transformers
The next challenge is to come up with something that is tolerable in terms of time. Simple vanilla implementation takes >1.5h for 1 epoch on GPU, which is way too long. To reduce the computation cost, I 

1) limit the lengths of story/summary: max_len1=1024, max_len2=128
2) implement the [FNet](https://arxiv.org/abs/2105.03824) for the encoder
3) the output of the encoder, called context in my code, is the same across all decoder layers. i.e. save the WQ, WK calculation
4) tested the [Performer](https://arxiv.org/abs/2009.14794), but due to short lengths, not quite useful.

These steps significantly reduce the training time down to ~10min/epoch.

Main codes:

fft_attention.py: FNet attention

vanilla_attention.py: Vanilla attention

performer_sharecontext_attention.py: Performer+share context attention; vanilla type of attention is also implemented in here

mixed_transformer_lm.py: The final transformer language model 

## Training
This is standard trainer I used many times before. I also tested the [Label-Distribution-Aware Margin Loss (LDAM)](https://arxiv.org/abs/1906.07413) but not so helpful. I am testing adding coverage contraint to alleviate the repetitive word problem, let's wait to see how it goes.

short note: ideally, one uses rouge score, but my code package will give warning after running a while, not sure why, so switch to bleu_score for a better training process.

test-train.ipynb:  training code

## Inference
I implement a topk=5 beamsearch, with max_queue = 20. The final rouge scores are rouge1:31, rouge2:11, rougeL:24. These are somewhat too high, which is likely due to the limit I put on the lengths of story/summary, i.e. a somewhat easier task is performed, compared to literature.

Some examples are:

Story:
by chris pleasance .published :03 :57 est ,6 january 2014 |updated :10 :39 est ,6 january 2014 .a family has escaped unharmed after the plane they were in had to make an emergency landing on a texas beach .father doyle thibodeaux ,his wife and their seven-year-old son were flying to houston on saturday when their single-engine aircraft was unable to switch to the last of four fuel tanks .their pilot ,brian himel ,tried to land at the nearby scholes international airport but was unable to make it .father doyle thibodeaux ,his wife ,and seven-year-old son escaped without injury after the plane they were in had a problem with its fuel tanks .instead he was forced to land the piper cherokee plane on stewart beach ,in galveston ,and luckily everyone escaped without injury .himel planned to fly the single-engine plane off the beach as soon as he could refuel .galveston police and parks officials had no immediate information sunday on plane .thibodeaux ,his wife and their son planned to continue their trip from patterson ,la .,to the houston area by car .elsewhere another piper pa-28 lost engine power while student pilot michael schwartz was taking two women for a tour of the statue of liberty on sunday .the quick-thinking amateur managed to safely land his plane on the major deegan expressway in the bronx without hitting any cars ,and saving the lives of both passengers .however ,today in aspen ,colorado ,a pilot was killed and two others injured as a private jet flipped over and burst into flames while trying to land .pilot brian himel was forced to make an emergency landing on stewart beach ,galveston ,after being unable to make it to nearby scholes international airport .co-pilot sergio carranza brabata ,from mexico ,died when the canadair cl- 600 skidded down the right side of runway at aspen-pitkin county airport ,flipped and exploded about 12 .30pm .the two injured men are also reportedly mexican pilots .they were rushed to aspen valley hospital ,with one in a critical condition .the aircraft was reportedly en route to the luxury ski resort town from toluca ,mexico via tuscon ,arizona .the plane is said to have attempted a few approaches before crashing in a ball of fire .emergency crews rushed to extinguish the flames and are currently investigating the cause of the crash .country singer leann rimes was among those who saw the explosion .she tweeted :'so sad !horrible plane crash we just saw happen at the aspen airport .'

Target:
plane forced to make emergency landing on stewart beach ,galveston .pilot was unable to switch to the smallest of the four fuel tanks .couple and their seven-year-old son escaped unharmed .<eos>

Predict:
father doyle thibodeaux ,his wife and seven ,were flying to houston on saturday .pilot ,brian himel ,tried to land piper cherokee plane on stewart beach ,in galveston ,and luckily everyone escaped without injury .pilot brian himel tried to land piper cherokee plane on stewart beach ,in galveston ,and luckily everyone escaped without injury .<eos>

test_rouge.ipynb: the inferencer is implemented here
