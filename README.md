# DeepIPCA

This repository implements IPCA models. 

Table of Contents
------------------------------

- [Empirical Results](#empirical-results)
- [Author](#author)

------

### Empirical Results

Here we report the results for the number of factor with the best validation Sharpe Ratio. 

- Sharpe Ratio

| Model          | Train | Valid | Test  |
|:--------------:|:-----:|:-----:|:-----:|
| Naive IPCA     | 1.04  | 1.17  | 0.47  |
| Iterative IPCA | 2.12  | 1.52  | 0.95  |
| Deep IPCA      |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/SR.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/SR.png" width="400" title="Iterative IPCA">
</p>

- Unexplained Variation

| Model          | Train | Valid | Test  |
|:--------------:|:-----:|:-----:|:-----:|
| Naive IPCA     | 0.99  | 0.99  | 0.99  |
| Iterative IPCA | 0.65  | 0.82  | 0.82  |
| Deep IPCA      |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/UV.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/UV.png" width="400" title="Iterative IPCA">
</p>

- Fama-McBeth Type Alpha (1e-03)

| Model          | Train | Valid | Test  |
|:--------------:|:-----:|:-----:|:-----:|
| Naive IPCA     | 1.45  | 3.50  | 4.00  |
| Iterative IPCA | 1.32  | 3.07  | 3.62  |
| Deep IPCA      |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/Alpha.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/Alpha.png" width="400" title="Iterative IPCA">
</p>

- Weighted Fama-McBeth Type Alpha (1e-04)

| Model          | Train | Valid | Test  |
|:--------------:|:-----:|:-----:|:-----:|
| Naive IPCA     | 1.00  | 5.51  | 1.70  |
| Iterative IPCA | 0.68  | 4.73  | 1.23  |
| Deep IPCA      |

<p align="center">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_naive/Alpha_weighted.png" width="400" title="Naive IPCA">
 <img src="https://github.com/LouisChen1992/DeepIPCA/blob/master/model/IPCA_Kelly/Alpha_weighted.png" width="400" title="Iterative IPCA">
</p>

### Author
- [Luyang Chen](https://github.com/louisChen1992)