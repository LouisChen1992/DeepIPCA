import os
import numpy as np
import tensorflow as tf

from src.utils import deco_print
from src.utils import sharpe

class ModelIPCA_FFN:
	def __init__(self, 
				individual_feature_dim, 
				tSize, 
				hidden_dims, 
				nFactor, 
				lr, 
				dropout,
				logdir, 
				dl, 
				force_var_reuse=False):
		self._individual_feature_dim = individual_feature_dim
		self._tSize = tSize
		self._hidden_dims = hidden_dims
		self._nFactor = nFactor
		self._lr = lr
		self._dropout = dropout
		self._logdir = logdir
		self._force_var_reuse = force_var_reuse
		
		self._load_data(dl)
		self._build_placeholder()
		with tf.variable_scope('Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()
		self._build_train_op()

	def _load_data(self, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			self._I_data = I[mask]
			self._R_data = R[mask]
			self._splits_data = mask.sum(axis=1)
			self._splits_np_data = self._splits_data.cumsum()[:-1]
			self._R_list_data = np.split(self._R_data, self._splits_np_data)

	def _build_placeholder(self):
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self._individual_feature_dim], name='IndividualFeature')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name='Return')
		self._F_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, self._nFactor], name='Factor')
		self._splits_placeholder = tf.placeholder(dtype=tf.int32, shape=[self._tSize], name='Splits')
		self._dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='Dropout')

	def _build_forward_pass_graph(self):
		with tf.variable_scope('NN'):
			h_l = self._I_placeholder
			for l in range(len(self._hidden_dims)):
				with tf.variable_scope('Layer_%d' %l):
					h_l = tf.layers.dense(h_l, self._hidden_dims[l], activation=tf.nn.relu)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

		with tf.variable_scope('Output'):
			self._beta = tf.layers.dense(h_l, self._nFactor)

		R_list = tf.split(value=self._R_placeholder, num_or_size_splits=self._splits_placeholder)
		beta_list = tf.split(value=self._beta, num_or_size_splits=self._splits_placeholder)
		F_list = tf.split(value=self._F_placeholder, num_or_size_splits=self._tSize)

		self._loss = 0
		for R_t, beta_t, F_t in zip(R_list, beta_list, F_list):
			R_hat_t = tf.squeeze(tf.matmul(beta_t, F_t, transpose_b=True), axis=1)
			self._loss += tf.reduce_sum(tf.square(R_t - R_hat_t))
		self._loss /= self._tSize

	def _build_train_op(self):
		optimizer = tf.train.AdamOptimizer(self._lr)
		self._train_op = optimizer.minimize(self._loss)

	def getBeta(self, sess):
		feed_dict = {self._I_placeholder:self._I_data,
					self._R_placeholder:self._R_data,
					self._splits_placeholder:self._splits_data,
					self._dropout_placeholder:1.0}
		beta, = sess.run(fetches=[self._beta], feed_dict=feed_dict)
		return beta

	def _step_factor(self, sess):
		beta = self.getBeta(sess)
		beta_list = np.split(beta, self._splits_np_data)
		F_list = []
		for R_t, beta_t in zip(self._R_list_data, beta_list):
			F_t = np.linalg.pinv(beta_t.T.dot(beta_t)).dot(beta_t.T.dot(R_t))
			F_list.append(F_list)
		return np.array(F_list)

	def _step_parameters(self, sess, F_data, maxIter=1024, tol=1e-06):
		old_variables = self.getParameters(sess)
		nIter = 0
		success = False
		loss_list = []
		while nIter < maxIter:
			feed_dict = {self._I_placeholder:self._I_data,
						self._R_placeholder:self._R_data,
						self._F_placeholder:F_data,
						self._splits_placeholder:self._splits_data,
						self._dropout_placeholder:self._dropout}
			_, loss = sess.run(fetches=[self._train_op, self._loss], feed_dict=feed_dict)
			loss_list.append(loss)
			new_variables = self.getParameters(sess)
			nIter += 1
			if self._max_norm_difference(old_variables, new_variables) < tol:
				success = True
				break
		if success:
			deco_print('Converged! ')
		else:
			deco_print('WARNING: Exceed maximum number of iterations! ')
		return loss_list

	def _max_norm_difference(self, v1_list, v2_list):
		tmp = 0.0
		for v1, v2 in zip(v1_list, v2_list):
			tmp = max(tmp, np.max(np.abs(v1 - v2)))
		return tmp

	def getParameters(self, sess):
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Model_Layer')
		return sess.run(trainable_variables)
