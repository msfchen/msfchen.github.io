---
layout: post
title: "Understanding the Family of Transformer Models. Part II - Long Sequence"
date: 2020-11-30
---
The input sequence length in most of the large-scale transformer-based language models, such as GPT, BERT, XLNet, and T5, are fixed at 512 tokens, which is not sufficient for many tasks involving longer context, such as document summarization, document-contexted question answering, document generation, and protein sequence analyses, etc. Splitting a long document into multiple segments of a fixed length may cause context fragmentation problem and reduce chance of learning long-range dependency. Moreover, the self-attention mechanism has both space and time complexity of $$O(d\times L^{2})$$, where $$d$$ is the embedding size and $$L$$ is the sequence length, making training and evaluating long sequences with transformer models very expensive. Many methods have been developed to improve transformer models for handling extra-long sequences. Some of the most prominent approaches in this line of study are reviewed here.

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

Another way to cover extra long sequence with reduced model capacity is to take advantage of the hierarchical structure of natural language by aggregating the outputs of lower layers elements to form the inputs to upper layers. This line of approach include hierarchical transformers, such as HIBERT, and hierarchical attention, such as BP-Transformer.

### **HIBERT**

Zhang et al., 2019<sup>[\[6\]](#ref6)</sup> introduced HIBERT (**HI**erachical **B**idirectional **E**ncoder **R**epresentations from **T**ransformers) specifically for extractive document summarization task. Extractive Summarization is usually modeled as a sentence ranking problem by selecting the most important sentences from a given document to form its summary. HIBERT is a hierarchical transformer encoder that stacks a document encoder on top of a sentence encoder. For pre-training, an additional sentence decoder is stacked on top the hierarchical encoder, as illustrated in Figure 1 below. For fine-tuning and inference, an additional linear classification layer is stacked on top the hierarchical encoder, as illustrated in Figure 2 below.
<p align="center"><img src="../../../assets/images/HIBERT.png"></p>
The sentence encoder, document encoder, and sentence decoder are all single stack transformer of the same size. There are two sizes: $$\mathrm{HIBERT}_{S}$$ has 6 layers, 8 attention heads, and hidden size 512; $$\mathrm{HIBERT}_{M}$$ has 6 layers, 12 attention heads, and hidden size 768. The length of each sentence is limited and truncated to be 50 words and each document is split into smaller documemnts with at most 30 sentences. Thus, each input document has at most 1500 words.

A document is represented as $$D=(S_{1}, S_{2},...,S_{\mid D\mid})$$, where $$S_{i}=(w_{1}^{i}, w_{2}^{i},...,w_{\mid S_{i} \mid}^{i})$$ is a sentence in $$D$$, $$w_{j}^{i}$$ a word in $$S_{i}$$, and $$w_{\mid S_{i} \mid}^{i}$$ an artifical End Of Sentence (EOS) token. A sentence is first mapped into embedding space $$E_{i}=(e_{1}^{i}, e_{2}^{i},...,e_{\mid S_{i} \mid}^{i})$$, where $$e_{j}^{i}=e(w_{j}^{i})+\mathrm p_{j}$$, where $$e(w_{j}^{i})$$ and $$\mathrm p_{j}$$ are the word and positional embeddings of $$w_{j}^{i}$$, respectively. Word embeddings are initialized randomly; positional embeddings use sinusoidal functions. The sentence encoder transforms $$E_{i}$$ into a list of hidden representations $$(h_{1}^{i}, h_{2}^{i},...,h_{\mid S_{i} \mid}^{i})$$. The representation, $$h_{\mid S_{i} \mid}^{i}$$, at the EOS token is taken as the aggregated representation of the sentence $$S_{i}$$. The positional embedding is incorporated into the final representation $$\hat h_{i}=h_{\mid S_{i} \mid}^{i}+\mathrm p_{i}$$ of $$S_{i}$$. The document encoder transforms $$(\hat h_{1}, \hat h_{2},..., \hat h_{\mid D\mid})$$ to the context sensitive sentence representations $$(d_{1}, d_{2},..., d_{\mid D\mid})$$ for document $$D$$.

In pre-training, 15% of sentences in each document are randomly selected, of which 80% have each of their tokens replaced by the [MASK] token; 10% remain the same; and 10% have each sentence replaced by a random sentence. The model is trained to predict the masked sentences. Given a document $$D=(S_{1}, S_{2},...,S_{\mid D\mid})$$, the masked document is denoted as $$\tilde{D}=(\tilde{S_{1}}, \tilde{S_{2}},..., \tilde{S_{\mid{D}\mid}})$$. Let $$K$$ denote the set of indicies of selected sentences in $$D$$, then the set of masked sentences in $$D$$ is $$M=\{S_{k}\mid k\in K\}$$, which is the target for prediction using $$\tilde{D}$$. The hierarchical encoders transform $$\tilde{D}$$ into $$(\tilde d_{1},\tilde d_{2},...,\tilde d_{\mid D\mid})$$. Then, the sentence decoder predicts masked sentence $$S_{k}=(w_{0}^{k}, w_{1}^{k},...,w_{\mid S_{k} \mid}^{k})$$ one word per step, where $$w_{0}^{k}$$ is an artificial [BOS] token. At the *jth* step, the model predicts $$w_{j}^{k}$$ given $$w_{0}^{k},...,w_{j-1}^{k}$$ and $$\tilde{D}$$. The probability of all masked sentences $$M$$ given $$\tilde D$$ is
$$p(M\mid\tilde D)=\prod\limits_{k\in K}\prod\limits_{j=1}^{\mid S_{k} \mid}p(w_{j}^{k}\mid w_{0:j-1}^{k},\tilde D)$$. The objective of the pre-training is to minimize the negative log-likelihood of all masked sentences given their corresponding documents.

In fine-tuning, extractive summarization is modeled as a sequence labeling problem, where the model takes in a document as sequence of sentences and assign a True or False label for each sentence to indicate whether a sentence should be included in the summary or not. Let $$D=(S_{1}, S_{2},...,S_{\mid D\mid})$$ and $$Y=(y_{1}, y_{2},...,y_{\mid D\mid})$$ denote a document and its corresponding labels, respectively. The hierarchical encoder transforms $$D$$ to the context dependent representations for all sentences $$(d_{1}, d_{2},..., d_{\mid D\mid})$$. The probability of the label of $$S_{i}$$ can be estimated using an additional linear projection and a softmax: $$p(y_{i}\mid D)=softmax(\mathrm{W}^{S}d_{i})$$ where $$\mathrm{W}^{S}\in \mathrm{\mathbb{R}}^{2\times d}$$. The fine-tuning objective is to minimize the negative log likelihood of all sentence labels given their corresponding documents.

The unlabled dataset GIGA-CM contains about 6.6M documents and about 2.9B words. Two labeled datasets, CNN/DailyMail and New Youk Times (NYT50, summary $$\geqslant$$ 50 words), are used for summerization experiments. The model is trained in three stages: (1) open-domain pre-training on GIGA-CM, (2) in-domain pre-training on CNNDM/NYT50, and (3) fine-tuning on CNNDM/NYT50. During inference, the top $$T$$ sentences, ranked by $$p(y_{i}\mid D)$$, are chosen as summary, where $$T$$ is tuned on the validation set.

The quality of summaries is evaluated by ROUGE scores. Using only the in-domain pre-training stage, the $$\mathrm{HIBERT}_{S}$$ (in-domain) significantly outperforms all previous models, including $$\mathrm{BERT}_{BASE}$$ that contains double number of model parameters ($$\mathrm{HIBERT}_{S}$$ 54.6M vs $$\mathrm{BERT}_{BASE}$$ 110M). With both open-domain and in-domain pre-training stages, $$\mathrm{HIBERT}_{S}$$ and $$\mathrm{HIBERT}_{M}$$ perform even better. Although $$\mathrm{HIBERT}_{M}$$ achieves new state-of-the-art performance, it still lags behind human.

### **BP-Transformer**

Ye et al., 2019<sup>[\[7\]](#ref7)</sup> introduce BP-Transformer (**B**inary **P**artitioning Transformer) that aggregates tokens of input sequence via binary partitioning and formulates an attention pattern of increasing span coverage with increasing distance, as illustrated below.
<p align="center"><img src="../../../assets/images/BPT1.png"></p>

The binary partitioning constructs a perfect binary tree, in which each leaf node corresponds to an input token, all internal nodes have two children, and all leaf nodes have the same depth. For a sequence with length $$n$$, there are $$2n-1$$ partitions, including $$n$$ token nodes and $$n-1$$ span nodes. Let $$u_{l,m}$$ denotes the $$mth$$ node at the level $$l$$, with $$l=0$$ for the level of token nodes. A span node $$u_{l,m}$$ represents a partition consisting of token nodes $$u_{0,2^{l}*m + 1},...,u_{0,2^{l}*(m + 1)}$$. Two types of edges are constructed: affiliated edges and contextual edges. The affiliated edges are directed edges from each of the containing token nodes to their span node. There are $$2^{l}$$ affiliated edges per span node, $$u_{0,2^{l}*m + i}\rightarrow u_{l,m}$$ for $$1\leq i \leq 2^{l}$$. The representation of a span node is computed by aggregating the representation of its containing token nodes. The contextual edges are directed edges from contextual nodes to token node $$u_{0,i}$$, where contextual nodes are $$k$$ (a hyper-parameter) nodes per level on the right-hand side starting at index $$p_{l}$$ at level $$l$$, where $$p_{0}=i+1$$ and $$p_{l}=p_{l-1}+k$$. The contextual nodes on the left-hand side are constructed similarly. Also, each node has a self-loop edge. The figure below illustrates the three types of edges in one graph self-attention layer. Lower-level span nodes cover local context and upper-level span nodes cover long-range context. The distances between any two token nodes are at most two edges. For a sequence of length $$n$$, the number of nodes and edges are $$O(2n)$$ and $$O(kn\log n/k)$$.  
<p align="center"><img src="../../../assets/images/BPT2.png"></p>
For a given node $$u$$, its neighbours $$A(u)$$ is set to be all its predecessor nodes. For each node $$v$$ in $$A(u)$$, a learnable representation $$r_{v,u}$$ of the relative positional difference between $$u$$ and $$v$$ are defined as (1) $$r_{v,u}=r^{self}$$ if $$v=u$$, (2) $$r_{v,u}=r_{j,i}^{left}$$ or $$r_{j,i}^{right}$$, if $$v$$ is the $$ith$$ left/right node to join the neighborhood set of $$u$$ at the $$jth$$ level, (3) $$r_{v,u}=r_{j}^{anc}$$, if $$u$$ is the ancestor of $$v$$ in the tree at level $$j$$. The positional encoding is $$R^{u}=concat(\{r_{v,u}\mid v\in A(u)\})$$. $$A^{u}=concat(\{h_{v}\mid v\in A(u)\})$$, where $$h_{v}$$ is the embedding of token $$v$$. $$Q_{i}^{u}=h_{u}W_{i}^{Q}$$, $$K_{i}^{u}=A^{u}W_{i}^{K}$$, $$V_{i}^{u}=A^{u}W_{i}^{V}$$. The attention by the $$ith$$ head for the node $$u$$ is $$head_{i}^{u}=softmax\bigg(\frac{Q_{i}^{u}(K_{i}^{u}+R^{u})^{\intercal}}{\sqrt d}\bigg)V_{i}^{u}$$. The graph self attention is $$[head_{1}^{u},...,head_{\mathrm h}^{u}]W^{O}$$, where $$\mathrm h$$ is the number of heads. The relative positional representations are shared across attention heads and each layer gets its own set of positional representations. For text classification and natural language inference tasks, the output is the root node in the final layer. For language modeling and machine translation tasks, the output is the representations of all the token nodes in the final layer.

BP-Transformer siginificantly outperforms vanilla Transformer on classification tasks using SST-5 and IMDB datasets. The best values of the hyperparameter $$k$$ (number of contextual nodes per level on each side) are 2 and 4 for SST-5 and IMDB datasets, respectively.

Character-level language modeling are evaluated on Enwiki8 and Text8 datasets. For fair comparison, all transformers use 12 layers, input length 512, embedding dimension 512, feed-forward dimension 2048, and k = 64. BP-Transformer matches the state-of-the-art performance at that time, achieved by the Adaptive-Span Transformer. The performance increases with the input context length up to 8192.

Encoder-decoder architecture is used for machine translation tasks. Document-level machine translation has document-level self attention and sentence-level inter-attention between encoder and decoder. On the document-level machine translation dataset IWSLT 2015 Chinese-to-English, BP-Transformer ($$k=4, l=64$$) siginificantly outperforms vanilla Transformer. On sentence-level machine translation dataset WMT14 English-to-German, BP-Transformer ($$k=4$$) outperforms vanilla Transformer. In general, $$k=4$$ appears to be the best setting for word-level NLP tasks on both small and large datasets.

BP-Transformer improves the time and space complexity of Transformers from $$O(d\times n^{2})$$ to $$O(d\times k\times n\log n/k)$$, where $$d$$, $$n$$, and $$k$$ are embedding dimension, input length, and number of contextual nodes per level per side, respectively. Increasing input length from 512 to 8192, BP-Transformer utilizes relatively constant GPU memory, but Transformer untilzes more GPU memory than BP-transformer and the difference increases as the input length increases. As for speed (tokens/sec), Transformers runs faster than BP-Transformer on short input length ($$\leq 1024$$); but Transformer becomes too slow for practical usage as the input length increases while the speed of BP-Transformer remains relatively steady.

## **Position-Based Sparse Attention**

The self-attention of the input sequences in the Transformer model is full attention, meaning each token attends to all other tokens or all left tokens in the same sequence. But each word in a sentence or document typically has depedency on only a small number of other words in the sentence or document, suggesting that full attention in language modeling leads to unnecessary cost in computation and storage and that reducing the attention scope of each query token to some pre-determined nearby locals may be a sufficiently good approxinmation to full attention. Some studies have exploited this strategy of enabling the transformer model to handle extra long input sequences by greatly reducing the self-attention cost per token.

### **Sparse Transformer**

Child et al., 2019<sup>[\[3\]](#ref3)</sup> introduce Sparse Transformers that use strided or fixed attention patterns, as illustrated below, in the decoder-only Transformer architecture. In the connectivity matrix below, rows and columns respresent output and input tokens, respectively. In (a), full attention in the Transformer for standard language modeling covers the lower-left half of the matrix. In (b) and (c), the strided and fixed attention patterns cover a small fraction of the attention connections.
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

Qiu et al., 2019<sup>[\[8\]](#ref8)</sup> took a simpler approach, named BlockBERT, to sparsify the attention matrices. BlockBERT sparsifies the attention matrices in a blockwise pattern and assigns different fraction of attention heads to different permutation of the blocks while all layers of the BlockBERT are treated in the same way. The primary benefit of BlockBERT is to reduce the memory and compute cost of the dot-product attention matrices from $$O(N^{2})$$ to $$O(N^{2}/n)$$ where $$N$$ and $$n$$ denote length of input seqnence and number of blocks, respectively, and each block matrix is of the size $$\frac{N}{n}\times\frac{N}{n}$$, as shown below.
<p align="center"><img src="../../../assets/images/blockBERT.png"></p>
The sparsity is achieved by elementwise multiplication of attention matrix with a masking matrix $$M\in\{0,1\}^{N\times N}$$ that sets corresponsing element of attention matrix to $$-\infty$$ when $$M_{ij}=0$$. The sparse pattern in M is defined by a permutation $$\pi$$ of $$\{1,2,...,n\}$$: $$M_{ij} = 1$$ if $$\pi\bigg(\big\lfloor\frac{(i-1)n}{N}+1\big\rfloor\bigg)=\big\lfloor\frac{(j-1)n}{N}+1\big\rfloor$$ and $$M_{ij} = 0$$ otherwise. Each masking matrix $$M_{i}$$ is determined by a permutation $$\pi_{i}$$. Permutations are generated by shifting one position, for example, (1, 2) and (2, 1) for $$n=2$$; (1, 2, 3), (2, 3, 1), and (3, 1, 2) for $$n=3$$, as in the Figure above. The identity permutations capture local dependency and others capture long-distance dependency.

Given block matrices $$Q=[Q_{1}^{\intercal} ... Q_{n}^{\intercal}]^{\intercal}, K=[K_{1}^{\intercal} ... K_{n}^{\intercal}]^{\intercal}, V=[V_{1}^{\intercal} ... V_{n}^{\intercal}]^{\intercal}$$, blockwise attention per head is defined as:

Blockwise-Attention$$(Q,K,V,M)=\bigg[softmax\bigg(\frac{Q_{1}K_{\pi(1)}^{\intercal}}{\sqrt d}\bigg)V_{\pi(1)}...softmax\bigg(\frac{Q_{n}K_{\pi(n)}^{\intercal}}{\sqrt d}\bigg)V_{\pi(n)}\bigg]$$.

Different attention head can use different masking matrices: $$head_{i}=$$Blockwise-Attention$$(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V}, M_{i})$$, where $$W_{i}^{Q}, W_{i}^{K}, W_{i}^{V}\in \mathrm R^{H\times d}$$ for $$H$$ number of hidden units and $$d$$ number of hidden units per head. Finally, attention from $$A$$ number of heads is defined as:

Blockwise-Multi-head-Attention $$(Q,K,V)=Concat(head_{1},...,head_{A})W^{O}$$, where $$W^{O}\in \mathrm R^{H\times H}$$.

The BlockBERT models follow the $$\mathrm{BERT_{BASE}}$$ settings: 12 layers, 12 heads, 768 hidden units, the same pre-training corpra and uncased word piece tokens. The number of tokens per batch is fixed at $$B\times N=131,072$$, where $$B$$ and $$N$$ denote batch size and input sequence length, respectively. Comparing to RoBERTa, BlockBERT significantly reduces memory usage and training time, with larger reduction for longer input sequence.

The model performance on downstream tasks are evaluated on seven different question answering datasets with different paragraph length distributions. SQuAD, NaturalQA, and HotpotQA consist of mostly short paragraphs (shorter than 512 tokens), while SearchQA and TriviaQA consist of longer paragraph with average length around 1,000 tokens. When the input sequence is longer than the configured $$N$$, it is split into a sliding window of size $$N$$ and stride 128. The input follow the format: [CLS]$$q_{1}...q_{t}$$[SEP]$$p_{1}...p_{s}$$[SEP], where $$q_{i}$$ and $$p_{i}$$ are tokens of question and paragraph, respectively. BlockBERT underperforms RoBERTa when $$N=512$$, but matches RoBERTa when $$N=1024$$. BlockBERT consistently outperform SparseBERT. Long sequence pre-training benefits long sequence fine-tuning. The heterogeneity of sequence length between pre-training and fine-tuning may hurt performance. BlockBERT with 2 blocks (n = 2) performs better than that with 3 blocks (n = 3). BlockBERT does achieve speedup and memory reduction during inference.

All optimal solutions assign considerable attention heads to block-diagonal matrices, or identity permuations. When $$n=2$$, the optimal number of heads for pre-training assigned to different permutations are 10:2 and 11:1 of (1, 2):(2, 1) for $$N=512$$ and $$N=1024$$, respectively. When $$n=3$$, the optimal number of heads for pre-training assigned to different permutations are 10:1:1 of (1, 2, 3):(2, 3, 1):(3, 1, 2) for both $$N=512$$ and $$N=1024$$. Pre-training performance and fine-tuning performance are correlated but not always consistent. The optimal number of heads for fine-tuning are 8:2:2.

### **Longformer**

All the sparse attention models reviewed above do not consider task-specific sparse attention for different downstream tasks. Beltagy et al., (2020)<sup>[\[5\]](#ref5)</sup> introduce Longformer that combines sparse local attention and few task-specific global attention. Longformer achieved new state-of-the-art results on a couple of question answering tasks and a document summarization task.
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
- [BP-Transformer](https://github.com/yzh119/BPT)
- [BlockBERT](https://github.com/xptree/BlockBERT)
- [Longformer](https://github.com/allenai/longformer)

## **References**

<a name="ref1">[1]</a> Dai, Z., Yang, Z., Yang, Y., Carbonell, J., Le, Q., and Salakhutdinov, R. (2019) [Transformer-XL: Attentive language models beyond a fixed-length context](https://arxiv.org/abs/1901.02860). arXiv preprint arXiv:1901.02860

<a name="ref2">[2]</a> Rae, J., Potapenko, A., Jayakumar, S., Hillier, C., and Lillicrap, T. (2019) [Compressive Transformers for Long-Range Sequence Modelling](https://arxiv.org/pdf/1911.05507.pdf). arXiv preprint arXiv:1911.05507

<a name="ref3">[3]</a> Child, R., Gray, S., Radford, A., and Sutskever, I. (2019) [Generating Long Sequences with Sparse Transformers](https://arxiv.org/pdf/1904.10509.pdf). arXiv preprint arXiv:1904.10509

<a name="ref4">[4]</a> Sukhbaatar, S., Grave, E., Bojanowski, P., and Joulin, A. (2019) [Adaptive Attention Span in Transformers](https://arxiv.org/pdf/1905.07799.pdf). arXiv preprint arXiv:1905.07799

<a name="ref5">[5]</a> Beltagy, I., Peters, M. E., and Cohan, A. (2020) [Longformer: The Long-Document Transformer](https://arxiv.org/pdf/2004.05150.pdf). arXiv preprint arXiv:2004.05150

<a name="ref6">[6]</a> Zhang, X., Wei, F., and Zhou, M. (2019) [HIBERT: Document Level Pre-training of Hierarchical Bidirectional Transformers for Document Summarization](https://arxiv.org/pdf/1905.06566.pdf). arXiv preprint arXiv:1905.06566

<a name="ref7">[7]</a> Ye, Z., Guo, Q., Gan, Q., Qiu, X., Zhang, Z. (2019) [BP-Transformer: Modelling Long-Range Context via Binary Partitioning](https://arxiv.org/pdf/1911.04070.pdf). arXiv preprint arXiv:1911.04070

<a name="ref8">[8]</a> Qiu, J., Ma, H., Levy, O., Yih, W., Wang, S., Tang, J. (2019) [Blockwise Self-Attention for Long Document Understanding](https://arxiv.org/pdf/1911.02972.pdf). arXiv preprint arXiv:1911.02972
