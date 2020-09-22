---
layout: post
title: "Knowledge Distillation"
date: 2020-09-20
---
Most of the recent breakthroughs from deep neural network models were accompanied with greatly increased number of parameters and amount of training data. For such large-scale models to be deployed into resource-constrained edge devices, model compression is required. Some authors have divided model compression techniques into four main categories.<sup>[\[1\]](#ref1)</sup> While most of these techniques involve in directly manipulating the weights of the hidden layers of the large models to make them smaller, the Knowledge Distillation technique manipulates the softmax layer to uncover additional information lost in standard softmax function and to learn more concise representations of the weights without significant loss in accuracy.

## What is Knowledge Distillation?

The term, Knowledge Distillation, was first introduced by Hinton et al., 2015.<sup>[\[2\]](#ref2)</sup> The knowledge here refers to the generalization ability on new data, learned by a large model, or ensemble of models. The distillation here refers to the machine learning process that transfers the knowledge of the large model to a small model that is called distilled model. The large-small model pair here is often referred to as teacher-student model pair in the literature.

### Temperature

A novel temperature term was introduced to the softmax function by Hinton et al., 2015<sup>[\[2\]](#ref2)</sup>, which was probably due to the resemblance between the softmax function and the Boltzmann distribution. The softmax temperature has nothing to do with the thermodynamic temperature.

| Softmax Function and Softmax Temperature *T* | Boltzmann Distribution and Thermodynamics Temperature *T* |
| :----: | :----:|
| $$q_i=\frac{e^{z_i/T}}{\sum_{j} e^{z_j/T}}$$ | $$p_i=\frac{e^{-\epsilon_i/kT}}{\sum_{j=1}^M e^{-\epsilon_j/kT}}$$ |
| *q<sub>i</sub>* and *z<sub>i</sub>* are the probability and logit, respectively, of the class *i*. | *p<sub>i</sub>* and *&epsilon;<sub>i</sub>* are the probability and energy, respectively, of the state *i*. *k* is the Boltzmann constant and *M* is the number of all states. |

For any given training instance, raising softmax temperature will decrease the probability of the class with the largest logit and increase the probabilities of all other classes, as shown in the chart below, an example of 6-class (A ~ F) classification task. These changes reveal some similarity structure over the data, such as class E is much closer than classes A ~ D to class F.  
<p align="center"><img src="../../../assets/images/prob_changes_by_T.png"></p>

### Distillation

In classification task, the ground truth class has label 1 and all other negative classes have label 0. These are called hard targets. The predicted label is 1 for the class with the highest predicted probability and 0 for all other classes. The main idea of Distillation is to use predicted softmax values from the large pre-trained model as the targets, called soft targets, for the small model using the same raised temperature (T > 1). This method can be significantly improved by also training the distilled model to produce correct hard targets at T = 1. The overall objective function can be a weighted average of the two objective functions. The training data is called transfer set that can be entirely different from the original training set of the large model. Using the original training set as the transfer set also works well.
<p align="center"><img src="../../../assets/images/KD_nn_architecture.png">
<br><em>Source: https://nervanasystems.github.io/distiller/knowledge_distillation.html</em></p>

Overall cross entropy objective function:
<p align="center">
$$CE(x,t)=\alpha(-t^2\sum_{i}\hat{y}_i(x,t)log y_i(x,t))+\beta(-\sum_{i}\bar{y}_ilog y_i(x,1))$$
</p>

$$\hat{y}_i(x,t)$$ is the predicted soft target for class *i* by the large model at *T=t*.

$$y_i(x,t)$$ is the predicted soft target for class *i* by the distilled model at *T=t*.

$$\bar{y}_i$$ is the known hard label for class *i*.

Since the magnitudes of the gradients produced by the soft targets scale as 1/$$t^{2}$$, it is important to multiply the distillation loss by $$t^{2}$$ to ensure that the
relative contributions of the hard and soft targets remain roughly unchanged if the temperature used for distillation is changed while experimenting with meta-parameters.

$$\alpha+\beta=1$$. According to Hinton et al., 2015<sup>[\[2\]](#ref2)</sup>, the best results were generally obtained by using a considerably lower $$\beta$$.

### Experiments on MINST

### Experiments on speech recognition

### Training on logits is a special case of distillation

1. Hinton paper
2. 2015 Do Deep Nets Really Need to be Deep? 
3. 2019_An Exploration on Temperature Term in Training Deep Neural Networks

### Mechanisms of konwledge distillation
1. 2020 Understanding and Improving Knowledge Distillation

## Example Applications
1. 2018_^_On-device neural language model based word prediction
2. 2019_TINYBERT DISTILLING BERT FOR NATURAL LANGUAGE UNDERSTANDING

## Quick Start Tools
1. PyTorch
    1. Neural Network Distiller (https://nervanasystems.github.io/distiller/index.html)?
    
2. Tensor Flow

## References

<a name="ref1">[1]</a> Cheng, Y., Wang, D., Zhou, P. & Zhang, T. (2018). Model compression and acceleration for deep neural networks: The principles, progress, and challenges. IEEE Signal Proc Mag 35(1):126–136

<a name="ref2">[2]</a> Hinton G , Vinyals O , Dean J . Distilling the Knowledge in a Neural Network. arXiv preprint arXiv:1503.02531
