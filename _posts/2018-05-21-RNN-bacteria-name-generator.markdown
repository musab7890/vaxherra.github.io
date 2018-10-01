---
layout: post
title: "Bacterial name generator with RNNs"
date: 2018-05-21
categories:
  - Data-Science
description: Create an RNN network and apply it to create novel bacterial genera name.
image: /_images/bosh/0014.png
image-sm: /_images/bosh/0014.png
---
In this post I'll cover how to build a Recurrent Neural Network (RNN) from scratch, as well as using existing Deep Learning (DL) frameworks to create new bacterial names from thousands of publicly available annotations.

In accompanying Jupyter notebooks I'll cover how to do this in:
1. Python and Numpy (painstaking implementation) <a href="https://github.com/vaxherra/vaxherra.github.io/blob/master/_files/bacterial_names/RNNs.ipynb">[see here]</a>
2. KERAS with Tensorflow backend (quick implementation) - <a href="https://github.com/vaxherra/vaxherra.github.io/blob/master/_files/bacterial_names/RNNs_KERAS.ipynb">[see here]</a>

I assume you have some knowledge of standard neural networks, are familiar with the ideas of propagation and backpropagation, loss and cost functions. I'll present how to construct an RNN unit, then an RNN network. I'll compute all the derivatives needed for backpropagation *'throught time'*. Finally I'll use the constructed networks and bacterial [SILVA database](https://www.arb-silva.de/) to extract bacterial genera names, and **sample** - generate novel names through our model.


# Motivation

For some time I was working on the human gut microbiome, analyzing alterations of compositionality and functional capability in disease (publication in progress). At the same time I was playing with some neural network architectures, like RNNs discussed here, and its ability to create novel sentences or words based on already acquired *"knowledge"*. In the era of Next Generation Sequencing (NGS), and especially Whole Metagenome Shotgun (WMS) sequencing it is now possible to study *'microbial dark matter'*, i.e. microbes that were previously uncharacterized, unknown mainly due to the hardships of cultivation or the specific geographic location, like some distant and exotic places (deep seas). I agree with [Dr. Murat Eren ](http://merenlab.org/2017/06/22/microbial-dark-matter/) that *'microbial dark matter'* is a term wrong on so many levels. As "*The dark matter in physics is a place holder for a not-yet-characterized form of matter that is distinct from ordinary matter*", whereas undiscovered microbes are just... uncharacterized. Hence, I won't be using this term again.

Some microbes [have been named after its discoverer](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_personal_names), some [after geographical names](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_geographical_names), and finally some [after institutions](https://en.wikipedia.org/wiki/List_of_bacterial_genera_named_after_institutions). Novel bacterial phyla, genera, species of strains are discovered now desipte limitations in its culturability. This is mainly achieved by computational methods: contig creation, genomic binning and further refinements based on single-copy core genes, redundancy measures etc... I though to myself, why not then come up with some bacterial name created by a neural network, that would create some novel names based on thousands of available genera name?

Here, I did just that. Since I quite enjoy playing with neural networks, I decidced to come up with this fun mini project. I decided also to carry out painstaking computations of backpropagation algorithm through RNNs. It is often neglected, as all existing frameworks do this automatically for you. It must have been Richard Feynman that said: *"You don't understand something completly if you have not built it yourself"*. If it was not him who said it, who cares? it is still a good advice, and I am going to apply it here.

I'll cover **some** RNN theory, carry out calculations of derivatives, propose RNN architecture, discuss how to train a model, and finally go over how to *sample* novel observations from the trained model. Some parts of my discussion may contain brief python code listings, however for a complete interactive examples please refer to the **jupyter notebooks** (links above). I encourage you to first read this post, then follow along jupyter notebook of your choice. That way you would have had built intuition by the time you read code implementations which are not as abundant in explanations. And as I mentioned earlier, feel free to chose a simple implementation with KERAS or a more complicated implementation with python.

# Input dataset

For this project I am going to use a [SILVA database](https://www.arb-silva.de/)  "*A comprehensive on-line resource for quality checked and aligned ribosomal RNA sequence data.*". In fact I'll only be focusing on taxonomy annotations for bacterial genera. Currently the latest release is numbered 132 and the `gzipped` file can be downloaded directly [here](https://www.arb-silva.de/fileadmin/silva_databases/release_132/Exports/SILVA_132_LSUParc_tax_silva.fasta.gz).

The previously mentioned `jupyter` notebooks provide code, a set of simple unix, `awk` and `sed` tricks piped together, showing how to reformat this `.fasta` file to come up with list of genera. The formatted file can be [downloaded here](/_files/bacterial_names/genera.txt). It contains **7901** genera names that are going to be used for our model training.




{% highlight bash %}

> head -10 genera.txt

Abelmoschus
Abies
Abiotrophia
Abolboda
Abortiporus
Abrahamia
Abroma
Abrus
Absidia
Abuta

{% endhighlight %}

For our purposes the input dataset is converted to lowercase, as it is better that our model doesn't learn something that is not useful for our task at hand.

# RNNs

## Introduction

For an extensive reading on the topic I direct you to the *resources* section at the end of this post. If you are already familiar with Recurrent Neural Networks  (RNNs), skip this section, as it only reviews what is special with RNNs as opposed to *feedforward networks*.

Standard neural models are often referred to as *feedforward networks* as each input, i.e. a data point or an image is treated as separate entity. The model has no "*memory"*, and is not a particularly good fit for processing sequence of data. An RNN is used to process sequences of information through iteration of each element of the data, like time-points, words or characters. It tries to *keep memory* of what came before.

Architecture of an RNN maintains an additional parameter, so called *state* that preserves information corresponding to what has been observed to far by the network. This additional parameter is really another variable retained from processing previous state. In our example in this post, the *'state'* corresponds to a word. This vague definition allows to construct many variations of RNNs, but for the purpose of this rather simplistic problem at hand, we'll consider the basic RNN architecture.


## Model Overview
In this post I am going to assume you are somewhat familiar with neural networks (NNs), since RNNs are variations on standard architecture. First, let me show you a nice RNN block diagram:

![a Basic RNN cell](/_images/bacterial_names/RNNcell.png)
<font size="2">Image source: Andrew Ng's "Sequence Models" <a   href="https://www.coursera.org/learn/nlp-sequence-models/">[ref].</a></font>

In this project I am going to generate words, so RNN model will inevitably work on characters. Let me briefly describe what each notation element stands for:

- \\( x^{\<t\>} \\) is an `t-th` element of sequence \\( X \\), i.e. a character. Input sequence has the formula \\( X = \{ x^{\<1\>},x^{\<2\>},...,x^{\<t-1\>},x^{\<t\>}  \} \\),

-  \\( W_{aa} \\), \\( W_{ax} \\), \\( W_{ya} \\): the weights of our RNN cell,

- \\( a^{\<t-1\>} \\), \\( a^{\<t\>} \\): are *hidden state* activations from a previous RNN cell (`t-1`) and for the next RNN cell (`t`),

- \\( \hat{y}^{\<t\>} \\): is a prediction, a probability of a given character at a time-step `t`, usually computed by a `softmax` function as we operate on one-hot encodings,

- \\( b_a \\): is a bias term


The rest is a set of additions and multiplications. Essentially the whole architecture is build upon this cell by means of repetition of RNN cells:

![a basic RNN architecture](/_images/bacterial_names/RNNarch.png)
<font size="2">Image source: Andrew Ng's "Sequence Models" <a   href="https://www.coursera.org/learn/nlp-sequence-models/">[ref].</a></font>

Forward pass through this network needs to compute \\( \hat{Y} \in ( \hat{Y}^{\<1\>}, ... , \hat{Y}^{\<T_x\>} ) \\) prediction vectors, hidden states \\( a \\), and store some of its computing results in *cache* that is needed for backpropagation. After forward pass, for our model we'll use a simple cross-entropy function as our model loss. For a single timestep it is defined as:

\\[  L_t(y_t,\hat{y}_t) = - y_t log(\hat{y}_t) \\]

and for an entire sequence, the cost function is just a sum of loss values:

\\[ L(y,\hat{y}) = - \Sigma_t y_tlog(\hat{y}_t) \\]


## Parameters initialization

The model, as shown on the image above, has many weights which need to be somewhat initialized before we begin training this network. These parameters are: \\( W_{aa},W_{ax}, W_{ya},b,b_y \\). As in any standard neural network setting these values to zero would not yield any "learning" results, as each pass and backpropagation would not adjust these weights. For this simplistic example a random initialization, between `0` and `0.01` would suffice for `W` parameters, and biases `b` could be initialized to zeros.

## Forward propagation

### Forward propagation through an RNN cell
Forward propagation is a set of repeated propagations through the RNN cell. Each RNN cell takes as input a set of parameters:
 \\( W_{aa},W_{ax}, W_{ya},b,b_y \\), previous state \\( a_{prev}\\) and current time-stamp, which is our character at particular position \\( x^{\<t\>}\\) (refer to the RNN architecture image above).

 Each RNN cell returns the next hidden state \\( a^{\<t+1 \>} \\) computed as an activation function, for example `tanh` applied to linear combination of weights and parameters, and current probability computed as a `softmax` (an activation function used for probability prediction) from current hidden state, and appropriate weights.

  \\[ a^{ \<t\>}  = tanh( W_{ax}x^{ \<t\>}  + W_{aa}a^{\<t-1\>}+b_a   )  \\]
  \\[ y^{ \<t\>}  = softmax( W_{ya}a^{ \<t\>}  +b_y   )  \\]

### Forward propagation through a network (of RNN cells)
I am repeating myself here, but for the sake of clarity: a forward pass through the network is just a repetition of single blocks, repeated \\( T_x \\) times (the length of the word).

As a result of this repeated forward propagation we should obtain \\( \hat{Y} \\), a vector of predictions (`softmax` probabilities), vector \\(a \\) of hidden states (from first to the last character). Our loss function is a standard cross-entropy function implemented as (for a single time-stamp):

\\[  L_t(y_t,\hat{y}_t) = - y_tlog(\hat{y}_t)  \\]

and for an entire RNN network architecture:

\\[ L(y,\hat{y}) = - \Sigma_t y_tlog(\hat{y}_t) \\]

## Backpropagation

Backpropagation in an RNN network is complicated, but not that hard. Essentially it is often said that deriving these computations by hand is the most *'complex'* thing in neural networks. Since many frameworks outsource this for their users, it has become neglected - and I don't blame anybody, it is really convenient. For a matter of practice however, in this section I am going to carry out all necessary calculations for the backpropagation through our RNN architecture.

If you don't feel like following these simple derivatives, skip to *'Backpropagation summary'* paragraph, where all necessary derivatives are computed.

### Backpropagation through a cell



Our model could be summarized with these three equations:

1. \\( L_t = - y^{\<t\>} log(\hat{y}^{\<t\>}) \\)
2. \\( \hat{y}^{\<t\>} = softmax ( W_{ya} a^{\<t\>} + b_y ) = softmax(z) \\)
3. \\( a^{\<t\>} = tanh  (W_{ax} x^{\<t\>} + W_{aa}a^{\<t-1\>} + b_a) \\)

It would also help to know how activation functions are defined:

- \\( softmax(z_j) =  \frac{e^z_j}{\Sigma^K_k e^z_k}\\) where \\(j=1,2,...k,k+1,...,K \\)

- \\( tanh(z) = \frac{sinh(z)}{cosh(z)} =  \frac{e^z - e^{-z}}{e^z + e^{-z}} \\)

With these we can calculate all the necessary derivatives. These calculations take considerate amount of space, so I put them under the link below:
### [Backpropagation calculations \[Link\] >> ](/statics/RNNbackprop)

If you followed the above link for calculations of backpropagation derivatives, then you already know where these values below come from, if not - then it's not really necessary for you to follow that.

Backpropagation through a single cell - final derivatives:
- \\(  \frac{\partial L_t}{\partial W_{ya}}  =  (\hat{y}^{\<t_l\>} -y^{\<t_l\>} ) \cdot a^{\<t\>}  \\)
- \\( \frac{\partial L_t}{\partial b_{y}} = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot \frac{\partial \hat{y}^{\<t\>}}{\partial z} \cdot 1 = \frac{\partial L_t}{\partial z} = \hat{y}^{\<t_l\>} -y^{\<t_l\>} \\)
- \\( \frac{\partial a^{\<t\>}}{\partial W_{aa}} = (1-a^{2\<t\>})a^{\<t-1\>} \\)
- \\( \frac{\partial a^{\<t\>}}{\partial b_a} = (1-a^{2\<t\>}) \\)
- \\( \frac{\partial a^{\<t\>}}{\partial W{ax}}  = (1-a^{2\<t\>}) \cdot x^{\<t\>} \\)
- \\( \frac{\partial L_t}{\partial W_{ya} } = (\hat{y}^{\<t_l\>} -y^{\<t_l\>}) W_{ya} \\)
- \\( \frac{\partial a^{\<t\>}}{\partial a_{\<t-1\>}} = (1-a^{2\<t\>})W_{aa} \\)

### Backpropagation through a network
After we've implemented our backward pass through the network for a single cell. Now we just have to repeat this for our network. The visualization below helps you to understand it better:

![RNN network - computing loss and backpropagation](/_images/bacterial_names/loss_compute.png)
[Image Source](https://www.coursera.org/learn/intro-to-deep-learning)

This backpropagation is called a backpropagation *'through time'*. If you think about it, we have to go in the opposite, i.e. reversed order to compute the desired gradient. We need to perform an iterative stepping in reversed order over each time step while **incrementing** (adding) the overall gradients: \\( \partial b_a, \partial W_{aa}, \partial W_{ax} \\).

#### Gradient clipping

After backpropagation and computing all the gradients we can make sure our gradients would not "explode", i.e. reach very high values compared to inputs. This phenomenon could make it very hard to effectively train our model. Here is some nice visualization of Loss function descending (gradient descent) with and without clipping:


![Gradient clipping](/_images/bacterial_names/clipping.png)
[Image Source](https://www.coursera.org/learn/intro-to-deep-learning)

Gradient clipping keeps the gradients values *'in check'*, that is between some arbitrary `min` and `max` values.

Training the model is performed in a standard fashion, compute loss, calculate gradient, update parameters, and iterate over # number of epochs through the data, trying to minimize this loss function. The key idea is then to get to know how to generate new observations...

## Sampling the model
Sampling is the process of generating novel *observations*. In our example new letters would construct a made-up bacterial genera name.

If our network is trained we can then pass a vector of zeros \\( \vec{0}\\) as input hidden state \\( a^{\<0\>} \\). Then we perform propagation through the first unit of our RNN network. So we obtain next hidden state \\(a^{\<t+1\>}\\) and prediction \\( \hat{y}^{\<t+1\>} \\) that represents the probabilities of each character `i`:

\\[ a^{\<t+1\>} = tanh(W_{ax}x^{\<t\>} + W_{aa}a^{\<t\>} + b) \\]
\\[ \hat{y}^{\<t+1\>}  = softmax( W_{ya}a^{\<t+1\>} + b_y  )  \\]

Having computed a vector of probabilities \\( \hat{y}^{<t+1>}\\) we now perform **sampling** procedure. We do not pick just the highest probability, this would in turn generate the same results each time for a given dataset. We do not want to pick our characters randomly, as results would become random, and all the architecture build would become useless. The key is to select (i.e. *sample*) from our \\( \hat{y}^{<t+1>} \\) distribution.

In other words, we'll pick the index `i` (remember that we have a one-hot encoded our alphabeth) with the probability encoded by the `i`-th index in \\(\hat{y}^{<t+1>} \\) matrix.

The final step is to **overwrite** the variable \\( x^{<t+1>}\\) with our predicted one-hot encoding \\(y^{\<t\>}\\) of selected/sampled index `i` from the previous step. This is represented as red arrow on the picture below:

![Gradient clipping](/_images/bacterial_names/sampling2.png)
[Image Source](https://www.coursera.org/learn/intro-to-deep-learning)

We'll continue this propagation until we reach end of the line character `\n`. Then our generation is finished, and we can print our result. If something goes wrong, then we additionally limit ourselves to an arbitrary number of character limit, for example `50`.


## Building a language models

The number of our network parameters is not dependent on the length of the input word. For training purposes we'll then just loop over one example at a time. Meaning, we'll forward propagate through RNN architecture, compute and clip gradients, update initial parameters with these computed gradients. Updating parameters means subtracting computed gradients that were multiplied by so called *learning rate*, an arbitrary and small value, like. `.0001`.


# Results
Using **7901** genera names from SILVA database I've trained this network and came up with a bunch of original bacterial names. Original in this context means that these names have not been present in our input database.

{% highlight bash %}

- Nitronella
- Sebacter
- Vetia
- Setinelfonax
- Vestaphylococcus
- Setonas
- Nembacterium
- Pioclococclus
- Detiptonus
- Frreptococcus
- Teeutomonas
- Fetiphylococcus
- Blunna
- Alococella
- Tantatum
- Cublia
- Palibacter
- Arstrosa
- Glymia
- Actoboctellibacterium
- Salanillus
- Sardaera

{% endhighlight %}

My presonal favourite is `Nitronella`, `Vetia` and `Frreptococcus`. You can inspect the `genera.txt` that I've obtained from the training dataset - the name is not there. Instead you can find:

`Nitrobacter, Nitrococcus, Nitrolancetus, Nitrosococcus, Nitrosomonas, Nitrosopelagicus, Nitrosospira, Nitrospina, Nitrospinae, Nitrospira, Nitrospirae, Nitrospirillum`.

 As for bacteria ending with `*ella`, there are 425 such observations, some of which are: `Volvariella, Nidorella, Veillonella, Weissella, Actinanthella, Truncocolumella, Traorella, Trichinella, Raoultella, Gloeotulasnella`.

Are these names useful? Perhaps. I think that names should be descriptive of origin or functionality, and it is really not my field of expertise to have a strong opinion here. But at the same time I feel that it might be a better alternative than naming a bacteria after a researcher.

Perhaps the next newly discovered bacteria will take its name from similar list? All things considered we're in the middle of AI revolution, and letting AI to name a new species will inevitably be regarded as a mark in OUR history.




# Resources
1. [Deep Learning Book by Aaron Courville, Ian Goodfellow, and Yoshua Bengio](https://www.deeplearningbook.org/)
1. [ Deep Learning with Python book by François Chollet ](https://www.manning.com/books/deep-learning-with-python)
1. [ National Research University Higher School of Economics: "Introduction to Deep Learning"](https://www.coursera.org/learn/intro-to-deep-learning)
1. [Dr Andrew Ng's "Sequence Models" course](https://www.coursera.org/learn/nlp-sequence-models/home/welcome)
