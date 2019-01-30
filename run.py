import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data.data_layer import DataInRamInputLayer
from src.model.model_IPCA_FFN import ModelIPCA_FFN

dl_train = DataInRamInputLayer('datasets/CharAll_na_rm_all_50_percent_train.npz')
dl_valid = DataInRamInputLayer('datasets/CharAll_na_rm_all_50_percent_valid.npz')
dl_test = DataInRamInputLayer('datasets/CharAll_na_rm_all_50_percent_test.npz')

model = ModelIPCA_FFN(46, 240, [], 4, 0.1, 1.0, '.', dl_train)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

F = np.random.randn(240, 4)

loss_list = []
for _ in range(10):
    loss = model._step_parameters(sess, F_data=F, maxIter=8)
    loss_list.append(loss)
    F = model._step_factor(sess)

plt.plot(np.concatenate(loss_list))
plt.show()
