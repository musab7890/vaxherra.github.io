---
layout: post
title: "Bacterial name generator with RNNs"
date: 2018-06-21
categories:
  - Data-Science
description: Create an RNN network and apply it to create novel bacterial genera name.
image: /_images/anomaly.png
image-sm: /_images/anomaly.png
---

In this post I'll cover how to build an RNN from scratch. In accompanying Jupyter notebook I'll cover how to do this in 1) Python and Numpy (painstaking implementation) and how to do it quickly in 2) KERAS with Tensorflow backend. I'll present how to construct an RNN unit, then an RNN network. I'll compute all the derivatives needed for backpropagation *'throught time'*. Finally I'll use the constructed networks and bacterial [SILVA database](https://www.arb-silva.de/) to extract bacterial genera names, and *sample*, that is generate novel names through build model. 



<font size="2">Cover photo source: ... <a target="_blank" href="">[ref].</a></font>


# Motivation

For some time I was working on the human gut microbiome, analyzing alterations of compositionality and functional capability in disease (publication in progress). At the same time I was playing with some neural network architectures, like RNNs discussed here, and its ability to create novel sentences or words based on already acquired *"knowledge"*. In the era of Next Generation Sequencing (NGS), and especially Whole Metagenome Shotgun (WMS) sequencing it is now possible to study *'microbial dark matter'*, i.e. microbes that were previously uncharacterized, unknown mainly due to the hardships of cultivation or the specific geographic location, like some distant and exotic places (deep seas). I agree with [Dr. Murat Eren ](http://merenlab.org/2017/06/22/microbial-dark-matter/) that *'microbial dark matter'* is a term wrong on so many levels, and shouldn't be used... So, I won't be using it again. I needed this term just to catch your attention.

Some microbes [have been named after its discoverer](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_personal_names), some [after geographical names](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_geographical_names), and finally some [after institutions](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_institutions). Novel bacterial phyla, genera, species of strains are discovered now desipte limitations in its culturability. This is mainly achieved by computational methods: contig creation, genomic binning and further refinements based on single-copy core genes, redundancy measures etc... I though to myself, why not then come up with some bacterial name created by a neural network, that would create some novel names based on thousands of available genera name? 

Here, I did just that. Since I quite enjoy playing with neural networks, I dediced to come up with this *fun project*. Also, I decided also to carry out painstaking computations of backpropagation algorithm through RNNs. It is often neglected, as all major existing frameworks do this automatically for you. It must have been Richard Feynman that said: *"You don't understand something completly if you have not built it yourself"*. If it was not him, who cares, it is still a good advice, and I am going to apply it here.

I'll cover **some** RNN theory, carry out calculations of derivatives, propose RNN architecture, discuss how to train a model, and finally go over how to *sample* novel observations from the trained model. Some parts of my discussion may contain brief python code listings, however for a complete interactive examples please refer to [jupyter notebook](FILL HERE). I encourage YOU first to read this post, then follow along jupyter notebook, so you would have had built intuition by the time you read code. 

# RNNs

## Overview
In this post I am going to assume you are somewhat familiar with neural networks (NNs), since RNNs are variations on standard architectures. First let me show you a nice RNN block diagram:

![a Basic RNN cell](/_images/bacterial_names/RNNcell.png)
<font size="2">Imageo source: Andrew Ng's "Sequence Models" <a target="_blank" href="https://www.coursera.org/learn/nlp-sequence-models/">[ref].</a></font>

RNN can work on either characters or words. In this project I am going to generate words, so RNN model will inevitabely work on characters. Let me briefly describe what each notation element stands for:

- \\( x^{<t>} \\) is an `t-th` element of sequence \\( X \\). Input sequence \\( X = \{ x^{<1>},x^{<2>},...,x^{<t-1>},x^{<t>}  \} \\),
- \\( W_{aa} \\), \\( W_{ax} \\), \\( W_{ya} \\): the weights of our RNN cell,
- \\( a^{<t-1>} \\), \\( a^{<t>} \\): are *hidden state* activation from previous RNN cell (`t-1`) and for the next RNN cell (`t`),
- \\( \hat{y}^{<t>} \\): is a prediction, a probability of a given character at a time-step `t`, usually computed by a `softmax` function as we operate on one-hot encodings,
- \\( b_a \\): is a bias term

The rest is a set of additions and multiplications. Essentialy the whole architecture is build upon this cell by means of repetition:

![a basic RNN architecture](/_images/bacterial_names/RNNarch.png)


Forward pass through this network needs to compute \\( \hat{Y} \\) (prediction) vectors, hidden states \\( a \\), and store some of its computing results in *cache* that is needed for backpropagation. After forward pass, for our model we'll use a simple cross-entropy function as a loss. For a single timestep it is defined as:

\\[  L_t(y_t,\hat{y}_t) = - y_tlog(\hat{y}_t) \\]

and for an entire sequence cost is just a sum of these values:

\\[ L(y,\hat{y}) = - \Sigma_t y_tlog(\hat{y}_t) \\]

## One-hot encodings

Input sequence at a given timestep `t`: \\( x^{<t>} \\) must come in a handy format for computations.


- padding


## Parameters initialization


## Backpropagation

Backpropagation in an RNN network is complicated, but not that hard. Essentialy it is often said that deriving these computations by hand is the most *'complex'* thing in neural networks. Since many frameworks outsource this for their users, it has become neglected - and I don't blame anybody, it really is convenient. For a matter of practice however, in this section I am going to carry out all necessary calculations for the backpropagation through our RNN architecture.

Since these calculations take much space, and would hide the *"big picture"*, please follow [this link](/helper_posts/RNN_backprop.markdown) for a complete set of computations. Below I reproduce the final derivative terms for convenience:

- test
- test
- test



# Resources


