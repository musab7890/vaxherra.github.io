---
layout: post
title: "Simple anomaly detection system"
date: 2018-05-15
categories:
  - Data-Science
description: Create a basic anomaly detection system in Python
image: /_images/anomaly.png
image-sm: /_images/anomaly.png
---
In this post I'll quickly go over a simple *anomaly detection system*: what is it, what are its motivations and how to build it. I'll use an example dataset of hypothyroid patients to look for anomalies. You can read this post, <a   href="https://github.com/vaxherra/vaxherra.github.io/blob/master/_files/anomaly_detection/anomaly.ipynb"><b><u>follow along a jupyter notebook </u></b></a> for code snippets or <a   href="/_files/anomaly_detection/anomaly.ipynb"><b><u>download</u></b></a> it for a local use. I obtained the dataset from [*Outlier Detection Dataset* website](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/), but you can download it <a   href="/_files/anomaly_detection/thyroid.mat"><b><u>here</u></b></a> if something happens to the source.  

<font size="1">Cover photo source: a cropped fragment from Pieter's Bruegel the Elder <i>'Netherlandish Proverbs'</i>. This painting is a list of proverbs and idioms, and this particular fragments means "To be able to tie even the devil to a pillow", i.e. Obstinacy overcomes everything <a   href="https://en.wikipedia.org/wiki/Netherlandish_Proverbs">[ref].</a></font>


## Motivation

### Overview

*Anomaly detection* is really an *outlier* detection problem. Given a certain set of observations we want to build some kind of understanding (model), so given a new example we can determine if it matches our previous observations or is not coming from the same distribution. An anomaly is different from standard classification problem due to the nature of our dataset. Often in anomaly detection we are given a dataset with *skewed classes*, i.e. we have much more *negative* (non-anomalous) that positive (anomalous) example. You can imagine a highly efficient production system, where most of our products are of good, acceptable quality, and are ready to be sold. Sometimes, however, we report a faulty product that cannot go into the market. Imagine that out of 100 thousand examples only 125 were *positive* (anomalous). So only 0.125% of examples are *positive*. Which is good for our production system, but makes it harder to use state of the art, 'data-hungry' machine learning algorithms based on neural nets (NN). NNs would probably not be able to properly learn what an *anomaly* means. We often say, there is an infinite number of ways something might go wrong, but usually, there is a limited number of ways to do something properly.

Let's say that we can automatically or semi-automatically collect some features from our products. Given the sheer number of features, it is often not possible to manually make sure our system or product it's working properly. But wait. Didn't I say that our example is going to include hypothyroid patients? Yes, indeed. For biological problems, this might be even harder, as biological systems usually are characterized by huge variability.


### Hypothyroid dataset overview
I've already mentioned that our dataset is related to the hypothyroid disease. I've obtained this dataset from [Outlier Detection Datasets](http://odds.cs.stonybrook.edu/thyroid-disease-dataset/) website. It comprises of 3772 subjects, only 93 subjects are characterized as hypothyroid (i.e. positive, anomalous) ( \\( 2.5\% \\) ). Each subject is attributed with six real-value features that we must use to build a model and predict whether it is or might be hypothyroid patient or not. In addition data has \\( \hat{Y} \in \{0,1 \} \\) labels, that state the *'ground truth'*, so we know whether a certain subject was actually hypothyroid or not. In fact, if you read the data described in the provided link, you can see that dataset merges two classes (normal and subnormal functioning) into one "normal class". Unfortunately, features are not named, so we actually don't know what each of them represents. But for learning purposes, this is enough. Ok, let's proceed to formulate a problem for our dataset.

### Problem formulation
Suppose we are working as a (clinical) data analyst for some medical organization close to a GP. One day we are given a moderate in size dataset of our patients. The staff has been collecting some six quantifiable features, be it symptoms or tests, that are indicative (to a limited extent) or suggest a hypothyroidism - "a condition in which the thyroid gland doesn't produce enough thyroid hormone". Our data also has a label stating whether a patient actually was hypothyroid.

We are not running a thyroid diagnosis and treatment center, but are a part of first contact team. For doctors, it would be nice to build some model that detects anomalies in patients, so further efforts might focus on specific organs or targeted diagnostics. Say, patient visits a GP, goes over symptoms and has a set of basic tests and measurements. Given our data, we can't really tell whether this patient is or is not hypothyroid, but rather whether there is some significant abnormality in a set of his results that, given small historical data, we may want to send him or her to check specifically for thyroid.

Our dataset is relatively small and skewed as most patients do not have hypothyroidism. This limits us directly in using "state of the art", everyone's favorite neural net classification system. However, we can try working with this data and construct an anomaly detection system.

## Building and testing a model

The idea is simple. We model each feature of our dataset by a multivariate Gaussian distribution: compute mean matrix ( \\( \mu \\)  ) and covariance matrix ( \\( \Sigma \\) ) on a training set:

\\[ \mu = \frac{1}{m} \Sigma^m_i x^{(i)} \\]

\\[ \Sigma = \frac{1}{m} \Sigma^m_i (x^{(i)}-\mu)\cdot(x^{(i)}-\mu)^T \\]

Mean matrix \\( \mu \\) contains a mean value for each feature `n`. Be sure to distinguish a summation sigma \\( \Sigma_i^m \\) from a covariance matrix sigma \\( \Sigma \\). This can be misleading, but just a bit. Then, given a new example \\( x^{i}_1,...,x^{i}_n \\) (with `n` features) we can compute its probability as defined by Multivariate Gaussian Distribution:

\\[ f_x(x^{i}) = \frac{  exp(- \frac{1}{2} (x^i-\mu)^T  \Sigma^{-1} (x^i-\mu)  ) }{  \sqrt{2\pi^n  \| \Sigma \|    }} \\]

Having computed probability of a given, new example, we can then decide based on that single number whether it is coming from our distribution or not. However, we should also determine a probability threshold \\( \epsilon \\) below which we consider an example as *anomalous*. But before we do that, we have to intelligently split our dataset.

We need to reasonably split the dataset for training, cross-validation and model testing. The idea is that for model training we use only negative examples, i.e. non-anomalous. Since we want to model each feature with a Gaussian distribution, it would be appropriate to `train` our model on negative examples and use a smaller portion of positive examples for hyperparameter tuning (\\( \epsilon \\) ) and model testing.

Table below shows how one might approach it for our hypothyroid dataset:

|Sets | # Negative examples | # Positive examples |
| ---| --- | --- |
| Train set| 2999 | 0 |
| Cross-validation set| 340 | 46 |
| Test set| 340 | 47 |


So train set is used to compute mean matrix \\( \mu \\) and covariance matrix \\( \Sigma \\). Cross-validation set is used to set an \\( \epsilon \\) probability threshold. How do we do this? We loop over an arbitrary number of possible probabilities, say one thousand or million points between the minimum and maximum probability obtained from a cross-validation set. Since our classes (anomalous vs non-anomalous) are skewed (there is disproportionately more negative examples that positives) we cannot use simple *accuracy* based on frequency, i.e. how many times our classifier was correct. Imagine that we predict \\( y=0 \\) all the time. Since the majority of our subjects are non-anomalous this would actually classify wrongly  93 out of 3772 and that would give us 97.53% accuracy. But this is just wrong. We do want to catch some anomalies if they likely occur, even if we might be wrong about them, as it is essentially better to send a non-hypothyroid patient for additional screening than to neglect actual hypothyroidism in a patient. We'd be happier even if we traded some of the precision for a recall.

Thus, we must operate using [`F1` score](https://en.wikipedia.org/wiki/F1_score) based on measure of precision and recall calculated using true positives (`tp`), false positives (`fp`) and false negatives (`fn`):


\\[ F_1 = \frac{2 \cdot precision \cdot recall}{precision + recall} \\]

\\[ precision =  \frac{tp}{tp+fp} \\]

\\[ recall = \frac{tp}{tp+fn}  \\]

Just a reminder. True positive refer to anomalous examples. If, for a given \\( \epsilon \\) over which we are iterating the computed probability \\( f_x^i < \epsilon \\) and our ground truth label \\( y=1 \\) then we can count an example as true positive. False negative would occur for the same true label, but we'd observe \\( f_x^i <> \epsilon \\). And finally a false positive would produce a small probability, below-given threshold \\( f_x^i < \epsilon \\), but in reality, the truth label says it's a non-anomalous example \\( y=0 \\).

Iterating over a set of \\( epsilon \\) values we choose the one that minimizes errors, i.e. maximizes our F1 score. We then use this \\( \epsilon \\) on the third portion of our data - *test set* and compute final precision, recall, and F1 scores.

If you follow <a   href="https://github.com/vaxherra/vaxherra.github.io/blob/master/_files/anomaly_detection/anomaly.ipynb"><b><u>jupyter notebook</u></b></a> for this post, then you observed that our final F1 score is \\( F1 \approx 0.73 \\) with \\( precision \approx 0.58 \\) and  \\( recall \approx 0.96 \\). Is this good?

## Final comment

Our model has a high recall, which means we very well identify all anomalies. So we are "good" at catching anomalies in patients when they occur, however when you look at our precision score, it is low. Precision essentially measures how well (precisely) our model identifies anomalies. I.e. our low score indicates that model will produce an "alert" even if some patient might not be hypothyroid. Is this good? Given the skewed dataset, and serious lack of positive examples I'd say it is helpful. When a new patients comes in, we perform a given set of basic tests, and a model predicts that he or she might be hypothyroid, we just sent this patient for a detailed set of tests that are more precise (or sensitive). This is not a serious decision on performing a complicated operation relying on our system. But at the same time, we make sure that we are catching \\( \approx 96\% \\) of anomalies, as our recall score indicates. Imagine you have a huge turnouver of patients in a first-contact clinic. Spending less time on manually looking at results and wondering whether they might be indicative of malfunctioning thyroid saves time for a GP. Also, at certain situations thus might *at hoc* suggest hypothyroid disease, and drive a GP to ask questions specific to symptoms (feeling increasingly tired, have dry skin, constipation and weight gain) or directly send a patient for a sensitive TSH test.

Oftentimes building a model and estimating accuracy is not enough. We need a model interpretation for a particular use. We have to analyze the nature of the problem and think about desired output. Here we have a standard precision and recall tradeoff, in which we are far better of with higher recall.
