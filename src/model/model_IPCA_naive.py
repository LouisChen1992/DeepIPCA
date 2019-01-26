import os
import numpy as np

from src.utils import deco_print
from src.utils import sharpe
from src.utils import decomposeReturn
from src.utils import UnexplainedVariation
from src.utils import FamaMcBethAlpha

class ModelIPCA_Naive:
	def __init__(self, individual_feature_dim, logdir):
		self._individual_feature_dim = individual_feature_dim
		self._logdir = logdir

	def _create_portfolio(self, R, C, mask):
		T = R.shape[0]
		R_masked = mask.astype(float) * R
		return np.array([R_masked[t].dot(C[t]) for t in range(T)])

	def _Markowitz(self, r):
		Sigma = r.T.dot(r) / r.shape[0]
		mu = np.mean(r, axis=0)
		w = np.dot(np.linalg.pinv(Sigma), mu)
		return w

	def _PCA(self, r, nFactors):
		Sigma = np.cov(r.T)
		eValues, eVectors = np.linalg.eig(Sigma)
		Lam = eVectors[:,:nFactors]
		w = self._Markowitz(r.dot(Lam))
		return Lam, w

	def train(self, dl_train, save=True):
		for _, (I_macro, I, R, mask) in enumerate(dl_train.iterateOneEpoch(subEpoch=False)):
			pTrain = self._create_portfolio(R, I, mask)
			for nFactors in range(1, self._individual_feature_dim+1):
				self._Lam, self._w = self._PCA(pTrain, nFactors)
				self._k = self._Lam.dot(self._w)
				if save:
					np.savez(os.path.join(self._logdir, 'model_%d.npz' %nFactors), lam=self._Lam, w=self._w, k=self._k)

	def loadSavedModel(self, nFactors):
		if os.path.exists(os.path.join(self._logdir, 'model_%d.npz' %nFactors)):
			tmp = np.load(os.path.join(self._logdir, 'model_%d.npz' %nFactors))
			self._Lam = tmp['lam']
			self._w = tmp['w']
			self._k = tmp['k']
			deco_print('Model Restored! ')
		else:
			self._Lam, self._w, self._k = None, None, None
			deco_print('WARNING: Model Not Found! ')

	def getFactors(self, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			p = self._create_portfolio(R, I, mask)
			return p.dot(self._Lam)

	def getSDFFactor(self, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			p = self._create_portfolio(R, I, mask)
			return p.dot(self._k)

	def getWeightWithData(self, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			return I[mask].dot(self._k)

	def calculateStatistics(self, dl):
		SR = sharpe(self.getSDFFactor(dl))
		beta = self.getWeightWithData(dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			R_hat, residual = decomposeReturn(beta, R, mask)
			UV = UnexplainedVariation(R, residual, mask)
			Alpha = FamaMcBethAlpha(residual, mask, weighted=False)
			Alpha_weighted = FamaMcBethAlpha(residual, mask, weighted=True)
		return (SR, UV, Alpha, Alpha_weighted)