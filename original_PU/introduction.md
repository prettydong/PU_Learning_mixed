# **Original PU Learning**

In this method , we use classical Two Step method (XL .Li 2002) and Gaussian Naïve Bayes .

(All dataset will be with two parts: 1. attributes 2. label(PN/PU))

## workflow

1. P->P and U->P:in this step ,we treat all unlabeled as Negative samples. And we call the dataset -> $D_0$
2. Use $D_0$ to train a Binary classifier $C_0$.
3. the Use $C_0$ to predict $D_0$.attributes and get $D_1$
4. Repeat step 2 & 3 until the numbers of P decrease.

