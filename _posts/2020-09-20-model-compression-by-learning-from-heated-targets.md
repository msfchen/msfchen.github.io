---
layout: post
title: "Model Compression by Learning from Heated Targets"
date: 2020-09-20
---
Most of the recent breakthroughs from deep neural network models were accompanied with increasing number of parameters and amount of training data. For such large-scale models to be deployed into capacity-limited edge devices, model compression is required. Some authors have divided model compression techniques into four categories.<sup>[\[1\]](#ref1)</sup> While most of these techniques involve directly manipulating the weights of the hidden layers of the large models to make them smaller, one line of studies have focused on training small models using various forms of predicted outputs of large models as targets to achieve a performance comparable to the large models. These studies include mimic learning<sup>[\[2\]](#ref2),[\[3\]](#ref3)</sup> and knowledge distillation<sup>[\[4\]](#ref4)</sup>.

- [What is knowledge distillation?](#what-is-knowledge-distillation)
    - [Temperature](#temperature)
    - [Distillation](#distillation)
    - [Experimental results](#experimental-results)
    - [Using logits as training targets](#using-logits-as-training-targets)
- [What is the underlying mechanism of knowledge distillation?](#what-is-the-underlying-mechanism-of-knowledge-distillation)
    - [Label Smoothing](#label-smoothing)
    - [Example re-weighting](#example-re-weighting)
    - [Prior of optimal geometry in logit layer](#prior-of-optimal-geometry-in-logit-layer)
    - [Order of contributions](#order-of-contributions)
- [Example Applications](#example-applications)
    - [Samsung Galaxy S7 keyboard language model compression](#samsung-galaxy-s7-keyboard-language-model-compression)
    - [TinyBERT - Distilling from Attention Layers](#tinybert---distilling-from-attention-layers)
- [Codes](#codes)
- [References](#references)

## **What is knowledge distillation?**

The term, knowledge distillation, was first introduced by Hinton et al., 2015.<sup>[\[4\]](#ref4)</sup> The knowledge here refers to the generalization ability on new data, learned by a large model, or ensemble of models. The distillation here refers to the machine learning process that transfers the knowledge of the large model to a small model, called distilled model, with raised temperature. The large-small model pair here is often referred to as teacher-student model pair in the knowledge distillation literature. Knowledge distillation technique is rooted in the mimic learning approach first demonstrated by Caruana el al, 2006<sup>[\[2\]](#ref2)</sup>, where large amount of synthetic data labelled by large, high-performing models were used to train small models to achieve high compression ratio and retain the performance of the large models. It showed that the generalization ability learned by a large, high-performing deep model can be learned by a small model. Recent advancements in this field have brought attention to the representations of training targets.

### **Temperature**

A novel temperature term was introduced to the softmax function by Hinton et al., 2015<sup>[\[4\]](#ref4)</sup>, which was probably due to the resemblance between the softmax function and the Boltzmann distribution. The softmax temperature has nothing to do with the thermodynamic temperature. It is a scaling factor of logit.

| Softmax Function and Softmax Temperature *T* | Boltzmann Distribution and Thermodynamics Temperature *T* |
| :----: | :----:|
| $$q_i=\frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$ | $$p_i=\frac{e^{-\epsilon_i/kT}}{\sum_{j=1}^M e^{-\epsilon_j/kT}}$$ |
| *q<sub>i</sub>* and *z<sub>i</sub>* are the probability and logit, respectively, of the class *i*. | *p<sub>i</sub>* and *&epsilon;<sub>i</sub>* are the probability and energy, respectively, of the state *i*. *k* is the Boltzmann constant and *M* is the total number of states. |

For any given training instance, raising softmax temperature will decrease the probability of the class with the largest logit and increase the probabilities of all other classes. For example, a training instance produces logits [-10, -5, 0, 5, 10, 15] for classes A ~ F, respectively, in a 6-class classification task, the chart below shows how their corresponding probabilities change as the temperature increases. These changes reveal some similarity structure over the data, such as class E is much closer than classes A ~ D to class F. Raising temperature can also be viewed as a way to amplify the differences between the probabilities of incorrect classes. Also note that if two training instances produce logits differing by a multiplication factor, say, [-2, -1, 0, 1, 2, 3] compared to the example above, their corresponding probabilities will be the same regardless of temperatures. This suggests that some useful information in the logits that are lost in the softmax may not be revealed even with raised temperatures.
<p align="center"><img src="../../../assets/images/prob_changes_by_T.png"></p>

### **Distillation**

In classification task, the ground truth class has label 1 and all other negative classes have label 0, namely one-hot labels. These are called hard targets. The predicted label is 1 for the class with the highest predicted probability, the argmax of the softmax outputs, and 0 for all other classes. The main idea of Distillation is to use predicted softmax values from the large pre-trained model as the targets, called soft targets, for the small model using the same raised temperature (T > 1). This method can be significantly improved by also training the distilled model to produce correct hard targets at T = 1. The overall objective function can be a weighted average of the two objective functions. The training data is called transfer set that can be entirely different from the original training set of the large model. Using the original training set as the transfer set also works well.
<p align="center"><img src="../../../assets/images/KD_nn_architecture.png">
<br><em>Source of the Diagram: https://nervanasystems.github.io/distiller/knowledge_distillation.html</em></p>

Overall cross entropy objective function:
<p align="center">
$$CE(x,t)=\alpha(-t^2\sum_{i}\hat{y}_i(x,t)log y_i(x,t))+\beta(-\sum_{i}\bar{y}_ilog y_i(x,1))$$
</p>

- $$\hat{y}_i(x,t)$$ is the predicted soft target for class *i* by the large model at *T=t*.
- $$y_i(x,t)$$ is the predicted soft target for class *i* by the distilled model at *T=t*.
- $$\bar{y}_i$$ is the known hard label for class *i*.
- Since the magnitudes of the gradients produced by the soft targets scale as 1/$$t^{2}$$, it is important to multiply the distillation loss by $$t^{2}$$ to ensure that the
relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters.
- $$\alpha+\beta=1$$. According to Hinton et al., 2015<sup>[\[4\]](#ref4)</sup>, the best results were generally obtained by using a considerably lower $$\beta$$.

### **Experimental results**
Experiments on MINST<sup>[\[4\]](#ref4)</sup>

| Model | Size | No. of Test Errors |
| :----: | :----:| :----:|
| Large | 2 hidden layers; 1,200 units/layer; strongly regularized | 67 |
| Small | 2 hidden layers; 800 units/layer; not regularized | 146 |
| Small | 2 hidden layers; 800 units/layer; distillation T = 20 | 74 |
| Small | 2 hidden layers; >= 300 units/layer; distillation T > 8 | similar to above |

Experiments on speech recognition<sup>[\[4\]](#ref4)</sup>

| Model | Size | Test Frame Accuracy | Word Error Rate |
| :----: | :----:| :----:| :----:|
| Single | 8 hidden layers; 2,560 units/layer; 14,000 labels; 85M parameters | 58.9% | 10.9% |
| Ensemble of 10 | ~ 850M parameters | 61.1% | 10.7% |
| Distilled Single | ~ 85M parameters; T=2 performed better than [1, 5, 10]; $$\alpha=\beta=0.5$$ | 60.8% | 10.7% |

### **Using logits as training targets**

Hinton et al., 2015<sup>[\[4\]](#ref4)</sup> showed mathematically that minimizing the difference between soft targets is equivalent to minimizing the difference between logits, if the temperature is high compared with the magnitude of the logits and the logits have been zero-meaned separately for each transfer case. Therefore, matching the logits of the teacher model is a special case of distillation.

Ba et al., 2014<sup>[\[3\]](#ref3)</sup> trained a single hidden layer student model on the logits, logarithms of predicted probabilities from teacher models (a deep CNN), as regression with L2 loss on logits and showed that it greatly outperformed the shallow model trained with original training data for the teacher model. They provided four possible reasons for the performance gain:

1. Teacher model corrected some label errors in original data.
2. Teacher model, through soft labels of logits, made complex regions in the mapping space easier to learn
3. Soft labels provide uncertainty across classes, which is more informative than hard labels with only one class as 1 and all other as 0.
4. The original targets may depend in part on features not available as inputs for learning. The targets from the teacher model are a function only of the available inputs; the dependence on unavailable features has been eliminated by filtering targets through the teacher model.

## **What is the underlying mechanism of knowledge distillation?**

Despite the popularity of knowledge distillation in practice, the underlying mechanism of how the mapping function of a larger representation can be learned by a much smaller representation has only begun to be investigated recently.

### **Label Smoothing**

Label smoothing is a very effective regularization technique, introduced by Szegedy et al., 2016<sup>[\[5\]](#ref5)</sup> to prevent the largest logit from becoming much larger than all others and therefore to generalize better. Given *K* classes and $$y_k$$ as the label of the *k-th* class, the smoothed label is defined as $$y_k^{LS}=y_k(1-\alpha)+\alpha/K$$, where $$\alpha$$ is the label smoothing parameter. The cross-entropy will use $$y_k^{LS}$$ instead of $$y_k$$.

Yuan et al., 2019<sup>[\[6\]](#ref6)</sup> compared the mathematical forms of the loss functions between using label smoothing and using knowledge distillation in training the student model and pointed out that the only difference is the artificially-imposed uniform distribution in the former and the teacher predicted distribution in the latter. Therefore, they argued that knowledge distillation is a learned label smoothing regularization, and label smoothing is an ad-hoc knowledge distillation with a teacher of random accuracy at T=1. At raised temperatures, teacher's soft targets distribution is more similar to the uniform distribution of label smoothing.

On the other hand, M&#252;ller et al., 2019<sup>[\[7\]](#ref7)</sup> showed that student models trained with label smoothing performed slightly worse than without; student models trained with teacher's provided targets (teachers trained with hard targets) all outperformed those trained with label smoothing. These results showed that knowledge distillation cannot be substituted by label smoothing. Surprisingly, students trained with better teachers (trained with label smoothing) is worse than students trained with ordinary teacher (trained without label smoothing). They showed that label smoothing impairs distillation by "erasing" the relative information between logits. This paper also showed that both label smoothing and temperature scaling implicitly calibrate learned models so that the confidences of their predictions are more aligned with the accuracies of their predictions.

Taking together, it is clear that raising temperature during distillation produces regularization effects on the logit layers, similar to label smoothing, which in turn reduces over-confidence problem, calibrates the model, and results in a better generalization ability. But the effects of label smoothing alone cannot explain all the observed results associated with knowledge distillation.

Tang et al., 2020<sup>[\[8\]](#ref8)</sup> argued that the regularization or calibration effect from smoothed teacher distribution cannot be considered as "knowledge" and they showed that teacher's "knowledge" are distilled through two other mechanisms: (1) example re-weighting and (2) prior knowledge of optimal geometry in logit layer.

### **Example re-weighting**

Tang et al., 2020<sup>[\[8\]](#ref8)</sup> introduced the term, gradient rescaling factor, $$w_i$$, as the ratio of logits gradient for knowledge distillation loss over logits gradient for hard targets loss. They showed both mathematically and experimentally that the gradient rescaling factor for ground truth class is positively correlated with the teacher's confidence of prediction for the ground truth class, which suggests that knowledge distillation will assign larger weights (in updating parameters) to training examples that are predicted to be the ground truth class by the teacher with higher confidence. Such automatic effect of example re-weighting according to teacher's prediction confidence is considered to be one of the "knowledges" for why the student models learn to generalize better when supervised by teacher's predicted outputs.

### **Prior of optimal geometry in logit layer**

Intuitively, logit $$z_k=h^{T}w_k$$ of the *k-th* class can be thought of as a measure of the squared Euclidean distance between the activations of the penultimate layer *h* and a weight template $$w_k$$, as $$\|h-w_k\|^{2}$$. Tang et al., 2020<sup>[\[8\]](#ref8)</sup> showed mathematically that at the optimal solution of the student, namely having the logits gradient for knowledge distillation loss as 0, for *T=1*, and for any two incorrect classes *i* and *j*, $$\|h-w_i\|^{2}<\|h-w_j\|^{2}$$ *iif* $$p_i>p_j$$. This suggests that the similarity relationship between incorrect classes is a part of the driving force to the optimal solution of student's output logit layer.

Tang et al., 2020<sup>[\[8\]](#ref8)</sup> also showed experimentally that using CIFAR-100 dataset containing 20 super-classes with 5 sub-classes each, classes within the same super-class have high correlations to each other in their teacher predicted probabilities at high temperature, but not at low temperature. This shows that teacher knowledge of class relationship is distilled to encourage hierarchical clustering.

### **Order of contributions**

To dissect the contributions of the two different mechanisms to the overall benefits of knowledge distillation, Tang et al., 2020<sup>[\[8\]](#ref8)</sup> devised synthetic teacher distributions $$\rho^{pt}$$ and $$\rho^{sim}$$ to isolate the effects of example re-weighting and optimal prior geometry of class relationship, respectively. The results showed that the benefits are in the order: example re-weighting > optimal prior geometry of class relationship > label smoothing.

The authors also discovered that adopting only the top-*k* largest values from the teacher distributions resulted in a better-quality student model, probably due to reduced noise in teacher distributions. The best *k* for the novel KD-topk method was 25%*K* in CIFAR-100 dataset and 50%*K* in ImageNet dataset.

## **Example Applications**

In practical model compression, knowledge distillation is often combined with other methods.

### **Samsung Galaxy S7 keyboard language model compression**

Yu et al., 2018<sup>[\[9\]](#ref9)</sup> distilled an ensemble of *K* teacher models into an LSTM-based student model, then applied three types of compression methods in sequence: shared matrix factorization, singular value decomposition (SVD), and 16-bit quantization to achieve state-of-art performance (at the time of its publication) in commercial mobile keyboard word prediction, by metrics of Keystroke Savings (KS) and Word Prediction Rate (WPR). The shared matrix factorization method reduced the total parameters in embedding and softmax layers by half. The SVD used *r* top singular values to achieve low rank compression (from $$m\times n$$ to $$m\times r$$ and $$r\times n$$), which was re-trained to restore accuracy. The 16-bit quantization reduced numerical precision of parameters from 32-bit to 16-bit. It compressed a 56.76MB model to 7.40MB, a compression ratio of 7.68, with only 5% loss in perplexity and achieved average prediction time of 6.47 ms.

### **TinyBERT - Distilling from Attention Layers**

Most recent advancements in natural language processing are based on the Transformer<sup>[\[10\]](#ref10)</sup> neural network that does not rely on sequential recurrent network. Instead, it is solely based on attention mechanisms to process all tokens at the same time and calculating attention weights between them. It consists of multiple layers, each of which has multiple attention heads. BERT<sup>[\[11\]](#ref11)</sup> was designed to pre-train a deep bidirectional transformer on large scale corpus of unlabeled data from general domain. The pre-trained BERT model can then be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of supervised tasks, such as question answering and language inference. Clark et al., 2019<sup>[\[12\]](#ref12)</sup> showed that substantial syntactic information is captured in BERT's attention. They showed that different attention heads specialize to different aspects of syntax and attention heads in the same layer tend to behave similarly.

In order to distill the knowledge of syntactic information from the attention layers of BERT model, Jiao et al., 2019<sup>[\[13\]](#ref13)</sup> devised a novel transformer distillation method that distilled from multiple layers of the teacher model and in both pre-training and task-specific training stages. The student model achieved 96% of the teacher performance on GLUE benchmark with 7.5x smaller size and 9.4x faster inference. It also significantly outperformed other earlier attempts to do knowledge distillation on BERT without distilling from attention layers.

## **Codes**

- PyTorch
    - [Neural Network Distiller by Intel AI Lab](https://github.com/NervanaSystems/distiller)
    - [A PyTorch Implementation of Knowledge Distillation](https://github.com/peterliht/knowledge-distillation-pytorch)
    - [TinyBERT](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT)
- TensorFlow
    - [knowledge distillation via TF2.0](https://github.com/sseung0703/Knowledge_distillation_via_TF2.0)
    - [Knowledge Distillation Methods with Tensorflow](https://github.com/sseung0703/KD_methods_with_TF)
    - [Knowledge Distillation - Tensorflow](https://github.com/DushyantaDhyani/kdtf)
- Keras
    - [Knowledge distillation with Keras](https://github.com/TropComplique/knowledge-distillation-keras)
    - [Knowledge distillation](https://github.com/tejasgodambe/knowledge-distillation)

## **References**

<a name="ref1">[1]</a> Cheng, Y., Wang, D., Zhou, P., Zhang, T. (2018). [Model compression and acceleration for deep neural networks: The principles, progress, and challenges](https://www.gwern.net/docs/ai/2018-cheng.pdf). IEEE Signal Proc Mag 35(1):126–136

<a name="ref2">[2]</a> Bucilu, C., Caruana, R., Niculescu-Mizil, A. (2006) [Model compression](http://www.niculescu-mizil.org/papers/rtpp364-bucila.rev2.pdf). In: Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ACM 535–541

<a name="ref3">[3]</a> Ba J., Caruana R. (2014) [Do deep nets really need to be deep?](https://papers.nips.cc/paper/5484-do-deep-nets-really-need-to-be-deep.pdf) In: Advances in neural information processing systems. 2654–2662

<a name="ref4">[4]</a> Hinton G , Vinyals O , Dean J. (2015) [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf). arXiv preprint arXiv:1503.02531

<a name="ref5">[5]</a> Szegedy C., Vanhoucke V., Ioffe S., Shlens J., Wojna, Z. (2016) [Rethinking the inception architecture for computer vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf). In: Proceedings of the IEEE conference on computer vision and pattern recognition, 2818–2826

<a name="ref6">[6]</a> Yuan, L., Tay, F. E., Li, G., Wang, T., Feng, J. (2019) [Revisit knowledge distillation: a teacher-free framework](https://arxiv.org/pdf/1909.11723.pdf). arXiv preprint arXiv:1909.11723

<a name="ref7">[7]</a> M&#252;ller, R., Kornblith, S., Hinton, G. (2019) [When does label smoothing help?](https://papers.nips.cc/paper/8717-when-does-label-smoothing-help.pdf) arXiv preprint arXiv:1906.02629

<a name="ref8">[8]</a> Tang, J., Shivanna, R., Zhao, Z., Lin, D., Singh, A., Chi, E. H., Jain, S. (2020) [Understanding and Improving Knowledge Distillation](https://arxiv.org/pdf/2002.03532.pdf). arXiv preprint arXiv:2002.03532

<a name="ref9">[9]</a> Yu, S., Kulkarni, N., Lee, H., Kim, J. (2018) [On-Device Neural Language Model based Word Prediction](https://www.aclweb.org/anthology/C18-2028.pdf). In: Proceedings of the 27th International Conference on Computational Linguistics: System Demonstrations, 128-131

<a name="ref10">[10]</a> Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A N., Kaiser, L., Polosukhin, I. (2017) [Attention is all you need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf). In: Advances in Neural Information Processing Systems, 6000–6010

<a name="ref11">[11]</a> Devlin, J., Chang, M., Lee, K., Toutanova, K. (2019) [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf?source=post_elevate_sequence_page---------------------------). arXiv preprint arXiv:1810.04805v2

<a name="ref12">[12]</a> Clark, K., Khandelwal, U., Levy, O., Manning, C. (2019) [What does BERT look at? An analysis of BERT’s attention](https://arxiv.org/pdf/1906.04341.pdf). arXiv preprint arXiv:1906.04341

<a name="ref13">[13]</a> Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., Liu, Q. (2019) [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/pdf/1909.10351.pdf). arXiv preprint arXiv:1909.10351v4
