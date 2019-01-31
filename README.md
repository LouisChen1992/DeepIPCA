# DeepIPCA

This repository implements IPCA models. 

Table of Contents
------------------------------

- [Empirical Results](#empirical-results)
- [Author](#author)

------

### Empirical Results

Here we report the results for the number of factor with the best validation Sharpe Ratio. 

- Number of Factors

| Model              |    |
|:------------------:|:--:|
| Naive IPCA         | 12 |
| Iterative IPCA     | 17 |
| Deep IPCA (Linear) | 14 |


- Sharpe Ratio

| Model              | Train | Valid | Test  |
|:------------------:|:-----:|:-----:|:-----:|
| Naive IPCA         | 1.04  | 1.17  | 0.47  |
| Iterative IPCA     | 2.12  | 1.52  | 0.95  |
| Deep IPCA (Linear) | 2.04  | 1.58  | 0.94  |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/SR.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/SR.png" width="400" title="Iterative IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_KellyWithFFN/SR.png" width="400" title="Deep IPCA (Linear)">
</p>

- Unexplained Variation

| Model              | Train | Valid | Test  |
|:------------------:|:-----:|:-----:|:-----:|
| Naive IPCA         | 0.99  | 0.99  | 0.99  |
| Iterative IPCA     | 0.65  | 0.82  | 0.82  |
| Deep IPCA (Linear) | 0.65  | 0.82  | 0.82  |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/UV.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/UV.png" width="400" title="Iterative IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_KellyWithFFN/UV.png" width="400" title="Deep IPCA (Linear)">
</p>

- Fama-McBeth Type Alpha (1e-03)

| Model              | Train | Valid | Test  |
|:------------------:|:-----:|:-----:|:-----:|
| Naive IPCA         | 1.45  | 3.50  | 4.00  |
| Iterative IPCA     | 1.32  | 3.07  | 3.62  |
| Deep IPCA (Linear) | 1.30  | 3.11  | 3.71  |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/Alpha.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/Alpha.png" width="400" title="Iterative IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_KellyWithFFN/Alpha.png" width="400" title="Deep IPCA (Linear)">
</p>

- Weighted Fama-McBeth Type Alpha (1e-04)

| Model          | Train | Valid | Test  |
|:--------------:|:-----:|:-----:|:-----:|
| Naive IPCA     | 1.00  | 5.51  | 1.70  |
| Iterative IPCA | 0.68  | 4.73  | 1.23  |
| Deep IPCA      | 0.68  | 4.79  | 1.25  |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/Alpha_weighted.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/Alpha_weighted.png" width="400" title="Iterative IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_KellyWithFFN/Alpha_weighted.png" width="400" title="Deep IPCA (Linear)">
</p>

### Author
- [Luyang Chen](https://github.com/louisChen1992)