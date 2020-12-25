---
layout: post
title: "Understanding the Family of Transformer Models. Part II - Long Sequence"
date: 2020-11-30
---
The input sequence length in most of the large-scale transformer-based language models, such as GPT, BERT, XLNet, and T5, are fixed at 512 tokens, which is not sufficient for many tasks involving longer context, such as document summarization, document-contexted question answering, document generation, and protein sequence analyses, etc. Splitting a long document into multiple segments of a fixed length may cause context fragmentation problem and reduce chance of learning long-range dependency. Moreover, the self-attention mechanism has both space and time complexity of $$O(L^{2})$$, where $$L$$ is the sequence length, making training and evaluating long sequences with transformer models very expensive. Many methods have been developed to improve transformer models for handling extra-long sequences. Some of the most prominent approaches in this line of study are reviewed here.

- [Extending Attention Span by Segment Recurrence](#extending-attention-span-by-segment-recurrence)
    - [Transformer-XL](#transformer-xl)
    - [Compressive Transformer](#compressive-transformer)
- [Hierarchically Aggregated Attention](#hierarchically-aggregated-attention)
    - [HIBERT](#hibert)
    - [BP-Transformer](#bp-transformer)
- [Position-Based Sparse Attention](#position-based-sparse-attention)
    - [Sparse Transformer](#sparse-transformer)
    - [Adaptive-Span Transformer](#adaptive-span-transformer)
    - [BlockBERT](#blockbert)
    - [Longformer](#longformer)
    - [Extended Transformer Construction](#extended-transformer-construction)
- [Content-Based Sparse Attention](#content-based-sparse-attention)
    - [Reformer](#reformer)
    - [Routing Transformer](#routing-transformer)
- [Generalized Attention](#generalized-attention)
    - [Performer](#performer)
    - [BigBird](#bigbird)
- [Codes](#codes)
- [References](#references)

## **Extending Attention Span by Segment Recurrence**

One way of encoding an arbitrarily long context into a fixed size representation is to split the long sequence into shorter segments of manageable sizes and then devise some mechanisms to propagate the information of previous segments to the next segments. Two examples of this approach are reviewed here: Transformer-XL and Compressive Transformer.

### **Transformer-XL**

Dai et al., 2019<sup>[\[1\]](#ref1)</sup> introduced segment-level recurrence mechanism into transformer architecture, in which the hidden states obtained in previous segments are fixed and cached so that they can be reused by the next segments recurrently. The information propagated through the recurrent mechanism can build up long-range relations between segments and resolve the context fragmentation problem, as illustrated in the figure below. The upper figure shows a 3-layer "vanilla model" that does not have information flowing across segments during training and has to calculate new segment from scratch for each new token during evaluation. The lower figure shows a 3-layer Transformer-XL that reuses, but not updates, the cached hidden states of previous segment as an extended context when the model processes the next new segment. In the experiments, the length of cache is equal to the segment length during training, and increased to multiple times of segment length during evaluation.
<p align="center"><img src="../../../assets/images/transformer_xl.png"></p>
Given $$s_{\tau}=[x_{\tau ,1},...,x_{\tau ,L}]$$ and $$s_{\tau +1}=[x_{\tau +1,1},...,x_{\tau +1,L}]$$ as two consecutive segments of length *L* and $$h_{\tau}^{n} \in \mathrm{\mathbb{R}}^{L\times d}$$ as *nth* layer hidden state sequence produced for the $$s_{\tau}$$, the *nth* layer hidden state sequence for the $$s_{\tau +1}$$ is produced as follows:

$$\tilde h_{\tau +1}^{n-1}=\left[ SG(h_{\tau}^{n-1})\circ h_{\tau +1}^{n-1} \right],$$

$$q_{\tau +1}^{n}, k_{\tau +1}^{n}, v_{\tau +1}^{n}=h_{\tau +1}^{n-1}\mathrm{W}_{q}^{\intercal}, \tilde h_{\tau +1}^{n-1}\mathrm{W}_{k}^{\intercal}, \tilde h_{\tau +1}^{n-1}\mathrm{W}_{v}^{\intercal},$$

$$h_{\tau +1}^{n}=\mathrm{TransformerLayer}(q_{\tau +1}^{n}, k_{\tau +1}^{n}, v_{\tau +1}^{n}),$$

where $$SG(\cdot)$$ denotes stop-gradient, $$[h_{u}\circ h_{v}]$$ denotes the concatenation
of two hidden sequences along the length dimension, and $$\mathrm{W}$$ denotes model parameters. The recurrent dependency between $$h_{\tau +1}^{n}$$ and $$h_{\tau}^{n-1}$$ shifts one layer per segment, different from the same layer recurrence in RNN. Consequently, the longest possible dependency length grows linearly with the number of layers and the length of segment.

The positional encoding in the original transfomer depends on token's absolute position in the input sequence. Tokens of the same absolute position within different segments will have the same positional encoding, which is not informative for learning across segments. To avoid this issue, the authors introduced a new relative positional encoding scheme that injects the relative distance, $$R_{i-j}$$, between query token, $$q_{i}$$, and key token, $$k_{j}$$, into the attention score, $$A_{i,j}^{rel}$$:

$$A_{i,j}^{rel}=E_{x_{i}}^{\intercal}W_{q}^{\intercal}W_{k,E}E_{x_{j}}+E_{x_{i}}^{\intercal}W_{q}^{\intercal}W_{k,R}R_{i-j}+u^{\intercal}W_{k,E}E_{x_{j}}+v^{\intercal}W_{k,R}R_{i-j},$$

where $$E_{x_{i}}$$ denotes embedding of token $$x_{i}$$, $$W_{k,E}$$ and $$W_{k,R}$$ denote weight matrices for producing the content-based key vectors and location-based key vectors, respectively, $$R\in \mathrm{\mathbb{R}}^{L\times d}$$ is a sinusoid encoding matrix without learnable parameters, $$u\in \mathrm{\mathbb{R}}^{d}$$ and $$v\in \mathrm{\mathbb{R}}^{d}$$ are trainable parameters, denoting query's positional weights for attending to keys' content and positions, respectively.

Effective context length studies show that Transformer-XL learns dependency that is 80% longer than RNNs and 450% longer than vanilla Transformers. Evaluation speed studies show that Transformer-XL achieves an up to 1,874 times speedup, over vanilla Transformers, during evaluation, due to the state reuse scheme. Transformer-XL achieved state-of-the-art performance on several benchmark tasks, involving long context. Although Transformer-XL is mainly designed to better capture longer-term dependency, it significantly outperforms vanilla Transformers on a task that mainly tests the ability of modeling only short-term dependency, suggesting the advantage of Transformer-XL is generalizable to modeling short sequences.

### **Compressive Transformer**

Rae et al., 2019<sup>[\[2\]](#ref2)</sup> introduced Compressive Transformer model that increases the extended context of the Transformer-XL by an additional Compressed Memory of past activations at each layer. The Compressive Transformer adopts the key ideas of the Transformer-XL, the segment-level recurrence and the relative positional encoding; but instead of discarding old activations that exceed cache length, it compresses and stores them in a secondary first-in-first-out (FIFO) Compressed Memory, as illustrated below.
<p align="center"><img src="../../../assets/images/compressive_transformer.png"></p>
The $$n_{s}$$, $$n_{m}$$, and $$n_{cm}$$ denote segment length, Memory (cached activations of previous segment(s)) length, and Compressed Memory length, respectively. "Sequence" in the figure above denotes the segment being processed. As the model moves to the next segment, its $$n_{s}$$ hidden activations are pushed into a fixed sized FIFO Memory (like the Transformer-XL). The oldest $$n_{s}$$ activations in Memory are evicted and processed by a compression function, $$f_{c}:\mathrm{\mathbb{R}}^{n_{s}\times d}\rightarrow \mathrm{\mathbb{R}}^{\lfloor\frac{n_{s}}{c}\rfloor\times d}$$, mapping the $$n_{s}$$ oldest memories to $$\lfloor\frac{n_{s}}{c}\rfloor$$ compressed memories that are then stored in the Compressed Memory. $$d$$ and $$c$$ denote the dimension of the hidden activations and the compression rate, respectively. A higher compression rate indicates more coarse-grained compressed memories. For a given layer $$i$$ at a given time step $$t$$, the extended context now is the concatenation of Compressed Memory and Memory, $$\left[ cm_{t}^{i}\circ m_{t}^{i} \right]$$. The table below compares the maximum possible context length and self-attention cost between Transformer-XL and Compressive Transformer. Assuming $$l$$, number of layers, and $$n_{s}$$ are the same, if Transfomer-XL's $$n_{m}$$ is $$2\times$$ of Compressive Transformer's $$n_{m}$$ (and $$n_{cm}$$), then, the two will have the same self-attention cost, but the Compressive Transformer will have $$2\times$$ larger context length when $$c=3$$.

| measure | Transformer-XL | Compressive Transformer |
| :----: | :----: | :----: |
| maximum context length | $$l\times n_{m}$$ | $$l\times (n_{m}+c\times n_{cm})$$ |
| self-attention cost | $$O(n_{s}^{2}+n_{s}n_{m})$$ | $$O(n_{s}^{2}+n_{s}(n_{m}+n_{cm}))$$ |

Four types of compression functions are compared: (1) max/mean pooling with the kernel and stride set to the compression rate c, (2) 1D convolution with kernel and stride set to c, (3) dilated convolutions, (4) most-used, where the memories are sorted by their average attention (usage) and the most-used are preserved and the least-used are removed. The convolutional
compression functions contain learnable parameters. The compression network is trained using some local auxiliary compression losses, to cope with vanishing gradient over long unrolls of very old memories. The Attention-Reconstruction Loss method is chosen, which reconstructs the content-based attention over memory, with content-based attention over the compressed memories. This is a lossy objective, as information that is no longer attended to can be discarded. The compression loss gradients are stopped from passing into the main network and there is no mixing between the Transformer objective and the compression objective. The 1D convolution compression function works the best.

To evaluate the long-range dependency learning ability, the author introduced a new benchmark dataset PG-19 that includes long text from books extracted from Project Gutenberg. The PG-19 contains 28,752 books in 11GB of text, more than $$2\times$$ the size of BookCorpus and Billion Word Benchmark. The average numbers of words per article are 69K, 3.6K, and 355 in PG-19, WikiText-103, and Penn Treebank, respectively. Training a Compressive Transformer ($$l=36, n_{s}=n_{m}=n_{cm}=512, c = 2$$) and a Transformer-XL ($$l=36, n_{s}=512, n_{m}=1024$$) on the GP-19 dataset obtained word-level test perplexity of 33.6 and 36.3, respectively. The Compressive Transformer also outperformed Transformer-XL on the standard character-level language modelling benchmark Enwiki8 and the closed-vocabulary word-level language modelling benchmark WikiText-103.

Monitoring the compression loss at each layer of the best-performing Compressive Transformer did not show any clear trend of compression cost increasing with higher layers in the network. Averaging the attention weight into eighteen buckets, six for each of the compressed memory, memory, and sequence revealed that most of the attention is placed on the current sequence with with a greater weight placed on earlier tokens of the sequence and that there is an increase in attention from the oldest activations stored in the regular memory to the activations stored in the compressed memory, indicating that older memories could be accessed more frequently than newer memories. The author also proposed a preferred optimization schedule: fast initial learning with frequent updates, and better generalisation near the end of training with less frequent updates instead of smaller learning rate.

Due to the additional space and time complexity, the Compressive Transformer is not suitable for the task that does not involve long-range reasoning.

## **Hierarchically Aggregated Attention**

### **HIBERT**

### **BP-Transformer**

## **Position-Based Sparse Attention**

The self-attention of the input sequences in the Transformer model is full attention, meaning each token attends to all other tokens or all left tokens in the same sequence. But each word in a sentence or document typically has depedency on only a small number of other words in the sentence or document, suggesting that full attention in language modeling leads to unnecessary cost in computation and storage and that reducing the attention scope of each query token to some pre-determined nearby locals may be a sufficiently good approxinmation to full attention. Some studies have exploited this strategy of enabling the transformer model to handle extra long input sequences by greatly reducing the self-attention cost per token.

### **Sparse Transformer**

Child et al., 2019<sup>[\[3\]](#ref3)</sup> introduced Sparse Transformers that use strided or fixed attention patterns, as illustrated below, in the decoder-only Transformer architecture. In the connectivity matrix below, rows and columns respresent output and input tokens, respectively. In (a), full attention in the Transformer for standard language modeling covers the lower-left half of the matrix. In (b) and (c), the strided and fixed attention patterns cover a small fraction of the attention connections.
<p align="center"><img src="../../../assets/images/sparse_transformers.png"></p>

Given that $$S_{i}$$ denotes the set of indices of the input tokens to which the *ith* output token attends, a connectivity pattern $$S=\{S_{1},...,S_{n}\}$$. Self attention on a sequence of input vectors $$X$$ can be represented as:

$$Attend(X, S)=(a(x_{i}, S_{i}))_{i\in \{1,...,n\}}, where$$

$$a(x_{i}, S_{i})=softmax\bigg(\frac{(W_{q}x_{i})K_{S_{i}}^{T}}{\sqrt d}\bigg) V_{S_{i}}, where$$

$$K_{S_{i}}=(W_{k}x_{j})_{j\in S_{i}} \quad \textrm{and} \quad V_{S_{i}}=(W_{v}x_{j})_{j\in S_{i}}$$

$$W_{q}, W_{k},$$ and $$W_{v}$$ denote the weight matrices that transform a given $$x_{i}$$ into a query, key, and value, respectively. $$d$$ is the inner dimension of the queries and keys. The attention output $$a$$ at the *ith* position is a sum of the values weighted by the scaled dot-product similarity of the keys and queries. In full self-attention of standard language modeling, $$S_{i}=\{ j:j \leq i\}$$. In factorized self-attention, $$p$$ separate attention heads are defined with different subset of indices, $$A_{i}$$, for attention. For the *mth* head, $$A_{i}^{(m)}\subset\{ j:j \leq i\}$$ and the $$S_{i}$$ in the equations above is substituted by $$A_{i}^{(m)}$$. The goal of the study is to restrict the size of $$A_{i}^{(m)}$$ according to $$\vert A_{i}^{(m)}\vert\propto \sqrt[p]{n}$$. Only $$p=2$$ is considered in the study, in which one head attends to the previous $$l$$ positions and the other head attends to every $$lth$$ positions.

In the strided attention pattern, Figure (b) above, the stride $$l$$ is chosen to be close to $$\sqrt{n}$$. $$A_{i}^{(1)}=\{ t, t+1, ...,i\}$$ for $$t=\max{(0, i-l)}$$ and $$A_{i}^{(2)}=\{ j:(i-j)\mod{l}=0\}.$$

In the fixed attention pattern, Figure (c) above, specific cells summarize previous locations and propagate that information to all future cells. $$A_{i}^{(1)}=\{ j:(\lfloor j/l\rfloor =\lfloor i/l\rfloor)\}$$, where the brackets denote the floor operation, and $$A_{i}^{(2)}=\{ j:j\mod{l}\in \{ t, t+1, ...,l\}\},$$ where $$t=l-c$$ and $$c$$ is a hyperparameter. The authors found that $$c\in \{ 8, 16, 32\}$$ for typical values of $$l\in \{ 128, 256\}$$ perform well.

Three different approaches are considered for integrating factorized self-attention: (1) using one attention pattern per block and interleaving them sequentially or at a ratio, (2) single merged head that attends to the positions that are attended by both factorized heads, and (3) multi-head producing attention in parallel and concatenating the results along the feature dimension.

Additional modifications from the original Transformer architecture are included to reduce memory usage and improve computation efficiency: (1) one-hot encoded positional encoding, instead of sine or cosine functions, (2) layer normalization is done before, instead of after, each sub-layer, (3) the states before and after each sub-layer as well as the final linear layer are check-pointed and stored in GPU memory, (4) attention weights and feedforward network activations are recomputed during the backpropagation, (5) the stride and fixed attention pattern are computed efficiently by slicing out sub-blocks from the query, key, and value matrices and computing the product in blocks, (6) weights are stored in single-precision floating-point, but activation and gradient are computed in half-precision to accelerate training.

A 30-layer fixed attention pattern Sparse Transformers was trained on the EnWik8 dataset, with a context length of 12,288, 8 heads, $$d=512$$, $$stride=128$$, $$c=32$$, and merged factorized attention heads. It matched the performance of a Transformer-XL model that contained more than double the number of parameters.

### **Adaptive-Span Transformer**

The Sparse Transformer applies the same pre-determined attention patterns on all attention heads in all layers. But Sukhbaatar et al., 2019<sup>[\[4\]](#ref4)</sup> show that some attention heads focus on recent history, while others attend uniformly to the whole available context. The authors devised an Adaptive-Span attention mechanism to learn the attention span of each head independently. Given a maximum allowed span $$S$$ and a learnable span $$z$$ of real value in $$[0, S]$$, a soft masking function $$m_{z}$$ that maps a distance $$x$$ to a value in $$[0, 1]$$ is defined as:
$$m_{z}(x)=\min\Big[\max\Big[\frac{\displaystyle 1}{\displaystyle R}(R+z-x), 0\Big], 1\Big]$$, where $$R$$ is a hyperparameter that controls its softness, as illustrated on the left below.
<p align="center"><img src="../../../assets/images/soft_mask_fn.png"></p>
Given a token position $$t$$ and its past token position $$r$$ in the span $$[t-S, t)$$, the attention weight from $$t$$ to $$r$$ is computed by applying the soft masking function to the softmax function:

$$a_{tr}=\frac{\displaystyle m_{z}(t-r)\exp(s_{tr})}{\displaystyle\sum_{q=t-S}^{t-1}m_{z}(t-q)exp(s_{tq})}$$, where $$s_{tr}=\mathrm x_{t}^{\top}\mathrm W_{q}^{\top}(\mathrm W_{k}\mathrm x_{r}+\mathrm p_{t-r})$$ is the similarity between tokens at positions $$t$$ and $$r$$, where $$\mathrm p_{t-r}$$ is the relative position embedding. $$L_{1}$$ regularization on the parameters $$z_{i}$$ for each attention head $$i$$ to the loss function:

$$L=-\log P(w_{1},..., w_{T})+\frac{\displaystyle\lambda}{\displaystyle M}\sum_{i}z_{i}$$, where $$\lambda >0$$ is the regularization hyperparameter and $$M$$ is the number of heads in each layer. The parameters $$z_{i}$$ are learned jointly with the rest of the parameters.

Character level language modeling on text8 and enwik8 datasets are used to compare the Adaptive Span model with standard Transformer and Transformer-XL models of similar sizes. Even with $$S=8192$$, the average span are $$z_{i}=314$$ and $$245$$ in small (12-layer) and large (24-layer) Adaptive Span model, respectively. The attention span of the Transformer and Transformer-XL are fixed at 512 and 3800, respectively. The large Adaptive Span model achieved state-of-the-art performance, at that time, on both datasets with fewer parameters and FLOPS (necessary for computing one-step prediction). As $$S$$ increasing from 256 to 4096, the average span and the FLOPS remain relatively constant in the Adaptive Span model, but increase in standard transformers, demonstrating that Adaptive Span models significantly reduce memoory usage and computation cost, in long context scenarios.

The attention span per head varies by the layer, as shown in the right figure above. In the 12-layer Adaptive Span model, the lowest 5 layers have small attention span; in contrast, some attention heads in the higher layers have very long spans, exceeding several thousand.

Attention span per input token, named Dynamic Attention Span, is also compared. At a time step $$t$$, the span parameter $$z_{t}$$ of an attention head is defined as a function of the input parameterized by a vector $$\mathrm v$$ and a scalar $$b$$, $$z_{t}=S\sigma(\mathrm v^{\top}x_{t}+b)$$. The $$z_{t}$$ is regularized similarly as the $$z_{i}$$ above and learned jointly with $$\mathrm v$$, $$b$$, and the rest of the parameters. The Dynamic Span model achieves the same performance as the Adaptive Span model with comparable average span on text8 dataset. The average dynamic span per token increases at the beginning of words and in the middle of composed words.

### **BlockBERT**

### **Longformer**

All the models reviewed above conducted character/word level language modeling to evaluate their performance on long text. It is unclear how they will perform on other language understanding tasks with long document inputs. Beltagy et al., (2020)<sup>[\[5\]](#ref5)</sup> introduce Longformer that combines sparse local attention and few task-specific global attention. Longformer achieved new state-of-the-art results on a couple of question answering tasks and a document summarization task.
<p align="center"><img src="../../../assets/images/longformer.png"></p>
The local attention pattern employs a sliding window of fixed-size $$w$$ surrounding each token, with $$w/2$$ tokens on each side, as shown in figure (b) above. This pattern has computation complexity of $$O(n\times w)$$, which scales linearly with $$n$$, the input sequence length. Assuming $$w$$ is fixed in all the layers of an $$l$$-layer transformer, the receptive field szie at the top layer is $$l\times w$$. Sliding window can be dilated by $$d$$ with gaps between attended tokens, figure (c) above, resulting in a receptive field of $$l\times d\times w$$. Increasing $$d$$ will increase the context length.

The global attention refers to few input positions pre-selected for a downstream task. A token with a global attention attends to all tokens of the input sequence, and all tokens of the input sequence attends to it. For example, [CLS] token is selected in the classification task, and all question tokens are selected in Question Answering task. Since the number of such tokens is small relative to and independent of $$n$$, the complexity of the combined local and global attention is still $$O(n)$$. Two sets of linear projection matrices are used to compute attention scores: $$Q_{s}, K_{s}, V_{s}$$ for sliding window attention, and $$Q_{g}, K_{g}, V_{g}$$ for global attention. These attention patterns can be plugged into any pre-trained transformer model without the need to change the model architecture.

For autoregressive language modeling, different dilated sliding window sizes and dilation are used at different layers, based on the findings of the Adaptive-Span Transformer. Small window and no dilation are used for lower layers and larger window and increasing dilation on only 2 heads are used for higher layers. The author found that a large number of gradient updates are needed to learn the local context first before learning to utilize longer context. Therefore, the model was trained in 5 phases: in the first phase, short sequence length (2,048) and small window size (32) are used; sequence length and window size are doubled in each subsequent phase. The Longformer-small (12-layer, 41M parameters) achieved new state-of-the-art performance on character-level language modeling using text8 and enwik8 datasets.

The Longformer was the first long-sequence transformer model being evaluated for a variety of language understanding tasks in the pre-training and fine-tuning scheme. The pre-training started from the RoBERTa released checkpoint, with minimal changes necessary to support Longformer’s attention mechanism. The sliding window attention used window size of 512, equivalent to RoBERTa's input sequence length. The model's input length is set as 4,096 tokens. The positional embeddings were initialized by copying the 512 positional embeddings from RoBERTa eight times. The data for pre-training included both short and long documents, with a distribution close to those used by RoBERTa. Masked language modeling was the goal of the pre-training. Two sizes of the model were trained: base model (12 layers, 8 heads, 512 hidden size) and large model (30 layers, 8 heads, 512 hidden size). Both were trained for 65K gradient updates.

Six datasets with average document length ranging from 506 to 6,589 were used for fine-tuning and evaluation tasks: three question answering datasets, WikiHop, TriviaQA, and HotpotQA; one coreference resolution dataset OntoNotes; and two document classification datasets IMDB and Hyperpartisan. Longformer-base consistently outperforms the RoBERTa-base on all six tasks, with larger gain on tasks that require long context such as WikiHop and Hyperpartisan. Longformer-large achieved new state-of-the-art at that time on WikiHop and TriviaQA by large margins. An ablation study for WikiHop showed that Longformer benefits from longer sequences, global attention, separate projection matrices for global attention, MLM pretraining, and longer training.

To study the impact of Longformer on sequence-to-sequence task, a variant, Longformer-Encoder-Decoder (LED), was developed. The LED was initialized from a pre-trained encoder-decoder Transformer, BART, and follow BART’s exact architecture in terms of number of layers and hidden sizes, except that the encoder's full self-attention was replaced by Longformer's local+global attention pattern. The positional embedding was extended from BART's 1K tokens to 16K tokens. The LED was evaluated on document summarization task using the arXiv summarization dataset. The 90th percentile of document lengths was 14.5K tokens. The encoder used local attention with window size 1,024 tokens and global attention on the first \<s\> token. LED was trained using teacher forcing on gold training summaries and used beam search at inference. LED achieved state-of-the-art results on arXiv.

### **Extended Transformer Construction**

## **Content-Based Sparse Attention**

### **Reformer**

### **Routing Transformer**

## **Generalized Attention**

### **Performer**

### **BigBird**

## **Codes**

- [Transformer-XL](https://github.com/kimiyoung/transformer-xl)
- [Adaptive-Span Transformers](https://github.com/facebookresearch/adaptive-span)
- [Longformer](https://github.com/allenai/longformer)

## **References**

<a name="ref1">[1]</a> Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., and Salakhutdinov, R. (2019) [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860). arXiv preprint arXiv:1901.02860

<a name="ref2">[2]</a> Rae, J., Potapenko, A., Jayakumar, S., Hillier, C., and Lillicrap, T. (2019) [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/pdf/1911.05507.pdf). arXiv preprint arXiv:1911.05507

<a name="ref3">[3]</a> Child, R., Gray, S., Radford, A., and Sutskever, I. (2019) [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf). arXiv preprint arXiv:1904.10509

<a name="ref4">[4]</a> Sukhbaatar, S., Grave, E., Bojanowski, P., and Joulin, A. (2019) [Adaptive Attention Span in Transformers](https://arxiv.org/pdf/1905.07799.pdf). arXiv preprint arXiv:1905.07799

<a name="ref5">[5]</a> Beltagy, I., Peters, M. E., and Cohan, A. (2017) [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf). arXiv preprint arXiv:2004.05150
