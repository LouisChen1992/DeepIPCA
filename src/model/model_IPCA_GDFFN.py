import os
import time
import numpy as np
import tensorflow as tf

from src.utils import deco_print
from src.utils import sharpe
from src.utils import decomposeReturn
from src.utils import UnexplainedVariation
from src.utils import FamaMcBethAlpha

class ModelIPCA_GDFFN:
	def __init__(self, 
				individual_feature_dim, 
				tSize, 
				hidden_dims, 
				nFactor, 
				lr, 
				dropout,
				logdir, 
				dl, 
				is_train=False,
				force_var_reuse=False):
		self._individual_feature_dim = individual_feature_dim
		self._tSize = tSize
		self._hidden_dims = hidden_dims
		self._nFactor = nFactor
		self._lr = lr
		self._dropout = dropout
		self._logdir = logdir
		self._logdir_nFactor = os.path.join(self._logdir, str(self._nFactor))
		self._is_train = is_train
		self._force_var_reuse = force_var_reuse
		
		self._load_data(dl)
		self._build_placeholder()
		with tf.variable_scope('Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()
		if self._is_train:
			self._build_train_op()

	def _load_data(self, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			self._I_data = I[mask]
			self._R_data = R[mask]
			self._mask_data = mask
			self._splits_data = mask.sum(axis=1)
			self._splits_np_data = self._splits_data.cumsum()[:-1]
			self._R_list_data = np.split(self._R_data, self._splits_np_data)

	def _build_placeholder(self):
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self._individual_feature_dim], name='IndividualFeature')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name='Return')
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
		F_list = []

		self._loss = 0
		for R_t, beta_t in zip(R_list, beta_list):
			F_t = tf.matmul(tf.linalg.inv(tf.matmul(beta_t, beta_t, transpose_a=True)),
							tf.matmul(beta_t, tf.expand_dims(R_t, axis=1), transpose_a=True))
			F_list.append(F_t)
			R_hat_t = tf.squeeze(tf.matmul(beta_t, F_t), axis=1)
			self._loss += tf.reduce_sum(tf.square(R_t - R_hat_t))
		self._loss /= self._tSize
		self._F = tf.concat([tf.transpose(tmp) for tmp in F_list], axis=0)

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

	def getFactors(self, sess):
		feed_dict = {self._I_placeholder:self._I_data,
					self._R_placeholder:self._R_data,
					self._splits_placeholder:self._splits_data,
					self._dropout_placeholder:1.0}
		F, = sess.run(fetches=[self._F], feed_dict=feed_dict)
		return F

	def getResidual(self, sess):
		beta = self.getBeta(sess)
		beta_list = np.split(beta, self._splits_np_data)
		F_list = list(self.getFactors(sess))
		residual_list = []
		for R_t, beta_t, F_t in zip(self._R_list_data, beta_list, F_list):
			residual_list.append(R_t - beta_t.dot(F_t))
		residual = np.zeros_like(self._mask_data, dtype=float)
		residual[self._mask_data] = np.squeeze(np.concatenate(residual_list))
		return residual

	def _Markowitz(self, r):
		Sigma = r.T.dot(r) / r.shape[0]
		mu = np.mean(r, axis=0)
		w = np.dot(np.linalg.pinv(Sigma), mu)
		return w

	def getMarkowitzWeight(self, sess):
		F = self.getFactors(sess)
		w = self._Markowitz(F)
		return w

	def getSDFFactor(self, sess, w):
		F = self.getFactors(sess)
		return F.dot(w)

	def evalLoss(self, sess):
		feed_dict_eval = {self._I_placeholder:self._I_data,
						self._R_placeholder:self._R_data,
						self._splits_placeholder:self._splits_data,
						self._dropout_placeholder:1.0}
		loss, = sess.run(fetches=[self._loss], feed_dict=feed_dict_eval)
		return loss

	def setLogdir(self, new_logdir):
		self._logdir = new_logdir
		self._logdir_nFactor = os.path.join(new_logdir, str(self._nFactor))

	def randomInitialization(self, sess):
		sess.run(tf.global_variables_initializer())
		deco_print('Random initialization')

	def loadSavedModel(self, sess):
		if tf.train.latest_checkpoint(self._logdir_nFactor) is not None:
			saver = tf.train.Saver(max_to_keep=128)
			saver.restore(sess, tf.train.latest_checkpoint(self._logdir_nFactor))
			deco_print('Restored checkpoint')
		else:
			deco_print('WARNING: Checkpoint not found! Use random initialization! ')
			self.randomInitialization(sess)

	def train(self, sess, model_valid, numEpoch=128, subEpoch=32):
		saver = tf.train.Saver(max_to_keep=128)
		if os.path.exists(self._logdir_nFactor):
			os.system('rm -rf %s' %self._logdir_nFactor)

		best_loss = float('inf')

		time_start = time.time()
		for epoch in range(numEpoch):
			deco_print('Doing Epoch %d' %epoch)
			for _ in range(subEpoch):
				feed_dict_train = {self._I_placeholder:self._I_data,
								self._R_placeholder:self._R_data,
								self._splits_placeholder:self._splits_data,
								self._dropout_placeholder:self._dropout}
				sess.run(fetches=[self._train_op], feed_dict=feed_dict_train)

			loss_train_epoch = self.evalLoss(sess)
			loss_valid_epoch = model_valid.evalLoss(sess)

			if loss_valid_epoch < best_loss:
				best_loss = loss_valid_epoch
				deco_print('Saving current best checkpoint')
				saver.save(sess, save_path=os.path.join(self._logdir_nFactor, 'model-best'))
			time_elapse = time.time() - time_start
			time_est = time_elapse / (epoch+1) * numEpoch
			deco_print('Epoch %d Train Loss: %0.4f' %(epoch, loss_train_epoch))
			deco_print('Epoch %d Valid Loss: %0.4f' %(epoch, loss_valid_epoch))
			deco_print('Epoch %d Elapse/Estimate: %0.2fs/%0.2fs' %(epoch, time_elapse, time_est))
			print('\n')

	def calculateStatistics(self, sess, w):
		SR = sharpe(self.getSDFFactor(sess, w))
		residual = self.getResidual(sess)
		R = np.zeros_like(self._mask_data, dtype=float)
		R[self._mask_data] = self._R_data
		UV = UnexplainedVariation(R, residual, self._mask_data)
		Alpha = FamaMcBethAlpha(residual, self._mask_data, weighted=False)
		Alpha_weighted = FamaMcBethAlpha(residual, self._mask_data, weighted=True)
		return (SR, UV, Alpha, Alpha_weighted)