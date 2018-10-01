---
layout: post
title: "Bacterial name generator with RNNs: Backpropagation derivatives"
date: 2018-05-21
categories:
  - Data-Science
description: Create an RNN network and apply it to create novel bacterial genera name.
image: /_images/bosh/0014.png
image-sm: /_images/bosh/0014.png
---
[<--- Go back to original post]({{ site.baseurl }}{% post_url 2018-10-01-RNN-bacteria-name-generator %})



#### a) `tanh` derivative
You can [look up](http://www.analyzemath.com/calculus/Differentiation/hyperbolic.html) the standard derivative of `tanh` (hyperbolic tangent) and it is equal to:

\\[ \frac{\partial tanh(z)}{\partial z} = sech^2(z) \\]

Another useful thing are the *hyperbolic identities* ([look it up here](http://math2.org/math/trig/hyperbolics.htm)), that state:

\\[ tanh^2(z) + sech^2(z) = 1 \Rightarrow  sech^2(z) = 1 - tanh^2(z)  \\]

combination of these two results in:

\\[ \frac{\partial tanh(z)}{\partial z}  = 1 - tanh^2(z) \\]

#### b) `softmax` derivative
Now let's focus on `softmax` derivative. The softmax of a parameter \\(z_j)\\, i.e: \\( softmax(z_j) \\) is really a 'normalized' value \\(z_j\\) that is exponentiated (`e` to the power of `z`) and then divided by all the possible values vector `z` might have. If we want to compute a derivative, we derivate over one value from \\(z\\) vector, i.e. \\(z_j\\). So, to put in in the derivatives terms:


\\[ \frac{\partial softmax(z_j))}{\partial z_i} = \frac{1}{\partial z_i} (\frac{e^z_j}{\Sigma^K_k e^z_k} )  \\]
  where \\(j=1,2,...k,k+1,...,K \\)

We have to consider two situations. One, where indices \\(i=j\\), the second where \\(i\neq j\\).

- if **\\(i=j\\)**, then the derivate of softmax will *look like* (just to present you):

\\[ \frac{\partial softmax(z_j))}{\partial z_i} =  \frac{1}{\partial z_i} (\frac{e^z_j}{e^z_j + \Sigma_{k, k\neq j}^K e^z_k })   \\]

and \\( \Sigma_{k, k\neq j}^K e^z_k\\) is essentially a *constant* part in calculating derivation, and for clarity I'll use \\(const.\\) to represent it:

\\[ \Sigma_{k, k\neq j}^K e^z_k \equiv const.  \\]

so there it goes:

\\[  \frac{\partial softmax(z_i))}{\partial z_i} =  \frac{1}{\partial z_i} (\frac{e^z_j}{e^z_j + const.})    \\]

I'll also use a basic rule of calculus - [quotient rule](http://tutorial.math.lamar.edu/Classes/CalcI/ProductQuotientRule.aspx), to recall:

\\[  \frac{1}{\partial x} (\frac{f(x)}{g(x)}) = \frac{ \frac{\partial f(x)}{\partial x} g(x) - f(x) \frac{\partial g(x)}{\partial x}   }{g(x)^2} \\]


Finally, we'll have:

\\[ \frac{\partial softmax(z_j))}{\partial z_i} =  \frac{ \frac{\partial e^z_j}{\partial z_i} (e^z_j + const.) - e^z_j \frac{\partial e^z_j + const.}{\partial z_i} }{ (e^z_j + const.)^2} = \frac{ e^z_j (e^z_j + const.) - e^z_j \cdot e^z_j}{ (e^z_j + const.)^2}  \\]

We can use our \\(softmax\\) definition to come up with:


\\[ \frac{\partial softmax(z_j))}{\partial z_i} =  softmax(z_j) - softmax^2(z_j) \\]
**if \\(i=j\\)**

Now for second possibility:
-  if **\\(j \neq i \\)**

\\[ \frac{\partial softmax(z_j))}{\partial z_i} =  \frac{1}{\partial z_i} (\frac{e^z_j}{e^z_i + \Sigma_{k, k\neq i}^K e^z_k })   \\]

Here we can treat \\(e^z_j \\) and \\( \Sigma_{k, k\neq i}^K e^z_k \\) as constant. We compute derivatives over \\(i\\) index, over \\(e^z_i\\). So this could be viewed as calculating a simple derivative:

\\[ \frac{\partial }{\partial x} (\frac{contst_1}{e^x + constant_2})  = const_1 \cdot  \frac{\partial }{\partial x} (\frac{1}{e^x + constant_2}) \\]

We can again use a quotient rule, but it is going to be much simpler:


\\[ ... = const_1 \cdot \frac{ 0 \cdot (e^x + constant_2) - 1 \cdot e^x }{(e^x + constant_2 )^2}  \\]

so:
\\[  \frac{\partial softmax(z_j))}{\partial z_i} = e^z_j \cdot \frac{1}{\partial z_i} (\frac{1}{e^z_i + \Sigma_{k, k\neq i}^K e^z_k})  =  ... \\]


\\[  ... =  e^z_j \cdot \frac{ \frac{1}{\partial z_i}(1) \cdot (e^z_i + \Sigma_{k, k\neq i}^K e^z_k) - 1 \frac{1}{\partial z_i}(e^z_i + \Sigma_{k, k\neq i}^K e^z_k)   }{(e^z_i + \Sigma_{k, k\neq i}^K e^z_k)^2 } = ... \\]


\\[ ... = e^z_j \cdot \frac{ 0  - e^z_i }{(e^z_i + \Sigma_{k, k\neq i}^K e^z_k)^2} = \frac{ - e^z_j \cdot  e^z_i }{(e^z_i + \Sigma_{k, k\neq i}^K e^z_k)^2}  = \frac{ - e^z_j \cdot  e^z_i }{( \Sigma_{k}^K e^z_k)^2} = - softmax(z_i) \cdot softmax(z_j)\\]

Alright, just to summarize our results and have a *clear* reference:


- if \\(i=j\\)
\\[ \frac{\partial softmax(z_j))}{\partial z_i} =  softmax(z_j) - softmax^2(z_j),  \text{     }\text{     }\text{          if } i=j \\]
- if \\(i \neq j \\)
\\[\frac{\partial softmax(z_j))}{\partial z_i} = - softmax(z_i) \cdot softmax(z_j),  \text{     }\text{     }\text{          if } i\neq j \\]


#### Calculating loss derivative with respect to \\(W_{ya}\\):
We need to calculate \\(\frac{\partial L_y}{\partial W_{ya}} \\)

Let's recall the functions that we'd be derivating over:
1. \\(L_t = - y^{\<t\>} log(\hat{y}^{\<t\>}) \\)
2. \\( \hat{y}^{\<t\>} = softmax (z ) \\)
3. \\( z =  W_{ya} a^{\<t\>} + b_y \\)

Following a simple chain rule we can write:

\\[ \frac{\partial L_t}{\partial W_{ya}} = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot \frac{\partial \hat{y}^{\<t\>}}{\partial z} \cdot  \frac{\partial z}{\partial W_{ya}} \\]

We can already see that: \\( \frac{\partial z}{\partial W_{ya}} = a^{\<t\>} \\). We also already computed derivative of $softmax$. Let's compute: \\( \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \\)

It is nice to remember that \\( \frac{1}{\partial x} (log(x)) = \frac{1}{x} \\). With that we are properly equipped to solve:

\\[ \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} = \frac{\partial}{\partial \hat{y}^{\<t\>}} (- y^{\<t\>} log(\hat{y}^{\<t\>}))  =  \frac{- y^{\<t\>}}{\hat{y}^{\<t\>}} \\]

Let's put everything here together:

\\[ \frac{\partial L_t}{\partial W_{ya}} = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot \frac{\partial \hat{y}^{\<t\>}}{\partial z} \cdot  \frac{\partial z}{\partial W_{ya}}  = \frac{- y^{\<t\>}}{\hat{y}^{\<t\>}} \cdot  \Big( ( softmax(z_i) - softmax^2(z_j)  ) + (- softmax(z_i)\cdot softmax(z_j)) \Big) \cdot a^{\<t\>} \\]

But first let's make sure we have proper notation. What is $\hat{y}^{<t>}$? It is our model's prediction. And this prediction is made with $softmax$ of course. Hence, we have to unify our notation:

\\[ \frac{\partial L_t}{\partial z} = \frac{- y^{\<t_i\>}}{\hat{y}^{\<t_i\>}} \cdot (\hat{y}^{\<t_i\>} - \hat{y}^{2\<t_i\>})  + \Sigma^K_{k,k\neq i} (\frac{- y^{\<t_k\>}}{\hat{y}^{\<t_k\>}}) \cdot (- \hat{y}^{\<t_k\>} \cdot \hat{y}^{\<t_i\>}) ...  \\]

Which is is simplified as

\\[ ... =  - y^{\<t_i\>} + y^{\<t_i\>}\hat{y}^{\<t_l\>} + \Sigma^K_{k,k\neq i} y^{\<t_k\>}\hat{y}^{\<t_i\>} \\]

We see the sum \\( \Sigma^K_{k,k\neq i} \\) that goes over every element except \\(k=l\\), but there is also one term \\(y^{\<t_i\>}\hat{y}^{\<t_l\>}\\) that adds this "except" term. We can thus simplify:

\\[ \frac{\partial L_t}{\partial z} = -y^{\<t_l\>} + \hat{y}^{\<t_l\>} \Sigma_k^K y^{\<t_k\>} \\]

So, just to recap. \\( \hat{y} \\) are predictions, they take values between 0 and 1, and sum up to one. But, \\(y\\) (without this funny hat) are *'ground truth'* labels. Essentially \\(y\\) is one-hot coded vector that has one `1` on the encoded position, and the rest are `0`s. It also sums to one obviously. Thus, we can further simplify:

$$ \frac{\partial L_t}{\partial z} = -y^{<t_l>} + \hat{y}^{<t_l>}  = \hat{y}^{<t_l>} -y^{<t_l>}  $$

So above is essentially prediction \\(\hat{y}\\) minus the true label. All this hassle for a simple result. Now plugging it into our desired derivative gives us our final result:

**\\[  \frac{\partial L_t}{\partial W_{ya}}  =  \frac{\partial L_t}{\partial z} \frac{\partial z}{\partial W_{ya}} =  (\hat{y}^{\<t_l\>} -y^{\<t_l\>} ) \cdot a^{\<t\>}  \\]**

We'll add this derivative, a.k.a **gradient** to each time-step \\(t\\) appropriately.

#### Calculating loss derivative with respect to bias \\( b_y \\)


This will be simple as we've already computed most of derivatives:

\\[ \frac{\partial L_t}{\partial b_{y}} = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot \frac{\partial \hat{y}^{\<t\>}}{\partial z} \cdot  \frac{\partial z}{\partial b_y} \\]

We essentially don't have **only the last component** which is equal to 1, as \\(z =  W_{ya} a^{\<t\>} + b_y\\), so \\(\frac{\partial z}{\partial b_y} = 1\\)

So we're left with :
\\[ \frac{\partial L_t}{\partial b_{y}} = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot \frac{\partial \hat{y}^{\<t\>}}{\partial z} \cdot 1 = \frac{\partial L_t}{\partial z} = \hat{y}^{\<t_l\>} -y^{\<t_l\>} \\]

which we know from previous calculations.

#### Calculating time gradient: \\( \frac{\partial a^{\<t\>}}{\partial W_{aa}} \\)
Let's recap our RNN cell:
![a basic RNN architecture](/_images/bacterial_names/RNNarch.png)

And our equations (I added a bit of helper notations, \\(p^{\<t\>}\\) and \\(z^{\<t\>}\\), in order to split them into several equations for the clarity):
1. \\(L_t = - y^{\<t\>} log(\hat{y}^{\<t\>})\\)
2. \\(\hat{y}^{\<t\>}  = softmax(z)\\)
3. \\(z^{\<t\>}=  W_{ya} a^{\<t\>} + b_y\\)
4. \\(a^{\<t\>} = tanh(p)\\)
5. \\(p^{\<t\>} = W_{ax} x^{\<t\>} + W_{aa}a^{\<t-1\>} + b_a\\)


Lets use (again) a chain rule for our equations:

\\[ \frac{\partial a^{\<t\>}}{\partial W_{aa}} = \frac{ \partial a^{\<t\>} }{\partial p}   \frac{\partial p}{\partial W_{aa}} \\]

We already know \\( \frac{ \partial a^{\<t\>} }{\partial p} \\) is just a derivative of `tanh` function we've computed before:
\\[ \frac{\partial tanh(z)}{\partial z}  = 1 - tanh^2(z) \\]

And \\( \frac{\partial p}{\partial W_{aa}} \\) is a simple derivative:
\\[ \frac{\partial p}{\partial W_{aa}}  = a^{\<t-1\>} \\]

Combining these we get:
\\[ \frac{\partial a^{\<t\>}}{\partial W_{aa}} = (1 - tanh^2(p^{\<t\>})  \cdot a^{\<t-1\>} = (1-a^{2\<t\>})a^{\<t-1\>} \\]

#### Calculating time gradient: \\( \frac{\partial a^{\<t\>}}{\partial b_a}  \\)

To calculate this derivative follow the chain rule (and numbered equations from previous paragraph):

\\[ \frac{\partial a^{\<t\>}}{\partial b_a} = \frac{\partial a^{\<t\>}}{\partial p} \cdot \frac{\partial p}{\partial b_a}   \\]

If you look above at `5.` equation, you see that \\( \frac{\partial p}{\partial b_a}=1 \\). This gives as automatically our result:

\\[ \frac{\partial a^{\<t\>}}{\partial b_a} = \frac{\partial a^{\<t\>}}{\partial p} \cdot \frac{\partial p}{\partial b_a} = (1-a^{2\<t\>}) \\]

#### Calculating input gradient: \\( \frac{\partial a^{\<t\>}}{\partial W{ax}} \\)
Most of calculations are already done. This doesn't need much explanation:

\\[ \frac{\partial a^{\<t\>}}{\partial W{ax}} = \frac{\partial a^{\<t\>}}{\partial p} \cdot \frac{\partial p}{\partial W_{ax}} = (1-a^{2\<t\>}) \cdot x^{\<t\>} \\]

As \\( \frac{\partial p}{\partial W_{ax}}  = x^{\<t\>} \\), and \\( \frac{\partial a^{\<t\>}}{\partial p} = (1-a^{2\<t\>})\\) which we know from previous calculations.

#### Calculating loss gradient with respect to hidden parameter \\( a^{\<t\>}\\)

Again, I am going to recall our equations for clarity:
1. \\(L_t = - y^{\<t\>} log(\hat{y}^{\<t\>})\\)
2. \\(\hat{y}^{\<t\>}  = softmax(z)\\)
3. \\(z^{\<t\>}=  W_{ya} a^{\<t\>} + b_y\\)
4. \\(a^{\<t\>} = tanh(p)\\)
5. \\(p^{\<t\>} = W_{ax} x^{\<t\>} + W_{aa}a^{\<t-1\>} + b_a\\)

\\[ \frac{\partial L_t}{\partial a_{\<t\>} }  = \frac{\partial L_t}{\partial \hat{y}^{\<t\>}} \cdot  \frac{\partial \hat{y}^{\<t\>}}{\partial z^{\<t\>} } \cdot  \frac{\partial z^{\<t\>} }{\partial a_{\<t\>}} \\]

The first two derivatives we have already solved.

\\[ \frac{\partial L_t}{\partial z}  = \hat{y}^{\<t_l\>} -y^{\<t_l\>}   \\]

\\[  \frac{\partial z^{\<t\>} }{\partial a^{\<t\>}}  = W_{ya} \\]

So we get:
\\[ \frac{\partial L_t}{\partial W_{ya} } = (\hat{y}^{\<t_l\>} -y^{\<t_l\>}) W_{ya} \\]

#### Calculating hidden state derivative over previous hidden state
And now, the final derivative:

\\[ \frac{\partial a^{\<t\>}}{\partial a^{\<t-1\>}}  = \frac{\partial a^{\<t\>} }{\partial p } \cdot \frac{\partial p}{\partial a^{\<t-1\>} } \\]

The first derivative is a derivative over `tanh` (already computed), the second is equal to \\(W_{aa}\\)

\\[ \frac{\partial a^{\<t\>}}{\partial a_{\<t-1\>}} = (1 - tanh^2(p^{\<t\>})  \cdot W_{aa} = (1-a^{2\<t\>})W_{aa} \\]

Ufff... derivatives are easy, but take time, patients and energy. No wonder major frameworks have this implemented it *"under the hood"* so people can only create computation graphs.

[<--- Go back to original post]({{ site.baseurl }}{% post_url 2018-05-21-RNN-bacteria-name-generator %})
