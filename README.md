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

| Model                          | Optimal Number of Factors |
|:------------------------------:|:-------------------------:|
| Naive IPCA                     | 12                        |
| Iterative IPCA                 | 17                        |
| Iterative Deep IPCA (Linear)   | 14                        |
| Iterative Deep IPCA (hpSearch) | 9                         |


- Sharpe Ratio

| Model                          | Train | Valid | Test  |
|:------------------------------:|:-----:|:-----:|:-----:|
| Naive IPCA                     | 1.04  | 1.17  | 0.47  |
| Iterative IPCA                 | 2.12  | 1.52  | 0.95  |
| Iterative Deep IPCA (Linear)   | 2.04  | 1.58  | 0.94  |
| Iterative Deep IPCA (hpSearch) | 1.88  | 1.78  | 0.93  |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/SR.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/SR.png" width="400" title="Iterative IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_KellyWithFFN/SR.png" width="400" title="Deep IPCA (Linear)">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_FFN/hpSearch/SR.png" width="400" title="Deep IPCA (hpSearch)">
</p>

- Unexplained Variation

| Model                          | Train | Valid | Test  |
|:------------------------------:|:-----:|:-----:|:-----:|
| Naive IPCA                     | 0.99  | 0.99  | 0.99  |
| Iterative IPCA                 | 0.65  | 0.82  | 0.82  |
| Iterative Deep IPCA (Linear)   | 0.65  | 0.82  | 0.82  |
| Iterative Deep IPCA (hpSearch) | 0.66  | 0.82  | 0.83  |

- Fama-McBeth Type Alpha (1e-03)

| Model                          | Train | Valid | Test  |
|:------------------------------:|:-----:|:-----:|:-----:|
| Naive IPCA                     | 1.45  | 3.50  | 4.00  |
| Iterative IPCA                 | 1.32  | 3.07  | 3.62  |
| Iterative Deep IPCA (Linear)   | 1.30  | 3.11  | 3.71  |
| Iterative Deep IPCA (hpSearch) | 1.37  | 3.21  | 3.74  |

- Weighted Fama-McBeth Type Alpha (1e-04)

| Model                          | Train | Valid | Test  |
|:------------------------------:|:-----:|:-----:|:-----:|
| Naive IPCA                     | 1.00  | 5.51  | 1.70  |
| Iterative IPCA                 | 0.68  | 4.73  | 1.23  |
| Iterative Deep IPCA (Linear)   | 0.68  | 4.79  | 1.25  |
| Iterative Deep IPCA (hpSearch) | 0.69  | 4.95  | 1.27  |

### Author
- [Luyang Chen](https://github.com/louisChen1992)