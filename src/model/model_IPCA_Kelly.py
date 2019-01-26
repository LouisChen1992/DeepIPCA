import os
import numpy as np

from src.utils import deco_print
from src.utils import sharpe
from src.utils import decomposeReturn
from src.utils import UnexplainedVariation
from src.utils import FamaMcBethAlpha

class ModelIPCA_Kelly:
	def __init__(self, individual_feature_dim, logdir):
		self._individual_feature_dim = individual_feature_dim
		self._logdir = logdir

	def _step_factor(self, R_list, I_list, Gamma, calculate_residual=False):
		f_list = []
		if calculate_residual:
			residual_list = []
		for R_t, I_t in zip(R_list, I_list):
			beta_t = I_t.dot(Gamma)
			f_t = np.linalg.pinv(beta_t.T.dot(beta_t)).dot(beta_t.T.dot(R_t))
			f_list.append(f_t)
			if calculate_residual:
				residual_list.append(R_t - beta_t.dot(f_t))
		if calculate_residual:
			return f_list, residual_list
		else:
			return f_list, None

	def _step_gamma(self, R_list, I_list, f_list, nFactors):
		tSize = len(f_list)
		A = np.zeros((self._individual_feature_dim * nFactors, self._individual_feature_dim * nFactors))
		b = np.zeros((self._individual_feature_dim * nFactors, 1))
		for R_t, I_t, f_t in zip(R_list, I_list, f_list):
			tmp_t = np.kron(I_t, f_t.T)
			A += tmp_t.T.dot(tmp_t)
			b += tmp_t.T.dot(R_t)
		Gamma = np.linalg.pinv(A).dot(b)
		return Gamma.reshape((self._individual_feature_dim, nFactors))

	def _Markowitz(self, r):
		Sigma = r.T.dot(r) / r.shape[0]
		mu = np.mean(r, axis=0)
		w = np.dot(np.linalg.pinv(Sigma), mu)
		return w

	def train(self, dl_train, initial_f_list, save=True, nFactorMax=46, 
		maxIter=1024, printOnConsole=True, printFreq=8, tol=1e-06):
		for _, (I_macro, I, R, mask) in enumerate(dl_train.iterateOneEpoch(subEpoch=False)):
			R_reshape = np.expand_dims(R[mask], axis=1)
			I_reshape = I[mask]
			splits = np.sum(mask, axis=1).cumsum()[:-1]
			R_list = np.split(R_reshape, splits)
			I_list = np.split(I_reshape, splits)

			for nFactors, initial_f in zip(range(1, self._individual_feature_dim+1), initial_f_list):
				if nFactors > nFactorMax:
					break
				self._Gamma = np.zeros((self._individual_feature_dim, nFactors))
				f_list = list(np.expand_dims(initial_f, axis=2))

				nIter = 0
				success = False
				while nIter < maxIter:
					Gamma = self._step_gamma(R_list, I_list, f_list, nFactors)
					f_list, _ = self._step_factor(R_list, I_list, Gamma)
					nIter += 1
					dGamma = np.max(np.abs(self._Gamma - Gamma))
					self._Gamma = Gamma
					if printOnConsole and nIter % printFreq == 0:
						deco_print('nIter: %d\t, dGamma: %0.2e' %(nIter, dGamma))
					if nIter > 1 and dGamma < tol:
						success = True
						break
				factors = np.squeeze(np.array(f_list), axis=2)
				self._w = self._Markowitz(factors)
				if save:
					np.savez(os.path.join(self._logdir, 'model_%d.npz' %nFactors), gamma=self._Gamma, w=self._w)
				
				if success:
					deco_print('Converged! (nFactors = %d)' %nFactors)
				else:
					deco_print('WARNING: Exceed maximum number of iterations! (nFactors = %d)' %nFactors)

	def loadSavedModel(self, nFactors):
		if os.path.exists(os.path.join(self._logdir, 'model_%d.npz' %nFactors)):
			tmp = np.load(os.path.join(self._logdir, 'model_%d.npz' %nFactors))
			self._Gamma = tmp['gamma']
			self._w = tmp['w']
			deco_print('Model Restored! ')
		else:
			self._Gamma = None
			deco_print('WARNING: Model Not Found! ')

	def getFactors(self, dl, calculate_residual=True):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			R_reshape = np.expand_dims(R[mask], axis=1)
			I_reshape = I[mask]
			splits = np.sum(mask, axis=1).cumsum()[:-1]
			R_list = np.split(R_reshape, splits)
			I_list = np.split(I_reshape, splits)
		f_list, residual_list = self._step_factor(R_list, I_list, self._Gamma, calculate_residual=calculate_residual)
		if calculate_residual:
			residual = np.zeros_like(mask, dtype=float)
			residual[mask] = np.squeeze(np.concatenate(residual_list))
			return np.squeeze(np.array(f_list), axis=2), residual
		else:
			return np.squeeze(np.array(f_list), axis=2), None

	def getSDFFactor(self, dl):
		factors, _ = self.getFactors(dl, calculate_residual=False)
		return factors.dot(self._w)

	def calculateStatistics(self, dl):
		SR = sharpe(self.getSDFFactor(dl))
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			_, residual = self.getFactors(dl, calculate_residual=True)
			UV = UnexplainedVariation(R, residual, mask)
			Alpha = FamaMcBethAlpha(residual, mask, weighted=False)
			Alpha_weighted = FamaMcBethAlpha(residual, mask, weighted=True)
		return (SR, UV, Alpha, Alpha_weighted)