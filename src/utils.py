import os
import numpy as np
import matplotlib.pyplot as plt

def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)

def sharpe(r):
	return r.mean() / r.std()

def decomposeReturn(beta, R, mask):
	R_reshape = R[mask]
	splits = np.sum(mask, axis=1).cumsum()[:-1]
	beta_list = np.split(beta, splits)
	R_list = np.split(R_reshape, splits)

	R_hat_list = []
	residual_list = []
	for R_t, beta_t in zip(R_list, beta_list):
		R_hat_t = beta_t.dot(R_t) / beta_t.dot(beta_t) * beta_t
		residual_t = R_t - R_hat_t
		R_hat_list.append(R_hat_t)
		residual_list.append(residual_t)
	R_hat = np.zeros_like(mask, dtype=float)
	residual = np.zeros_like(mask, dtype=float)
	R_hat[mask] = np.concatenate(R_hat_list)
	residual[mask] = np.concatenate(residual_list)
	return R_hat, residual

def UnexplainedVariation(R, residual, mask):
	N_t = np.sum(mask, axis=1)
	UV = np.mean(np.square(residual).sum(axis=1) / N_t) / np.mean(np.square(R * mask).sum(axis=1) / N_t)
	return UV

def FamaMcBethAlpha(residual, mask, weighted=False):
	T = mask.shape[0]
	T_i = np.sum(mask, axis=0)
	if weighted:
		Alpha = np.mean(np.square(residual.sum(axis=0) / T_i) * T_i) / T
	else:
		Alpha = np.mean(np.square(residual.sum(axis=0) / T_i))
	return Alpha

def calculateAllStatistics(model, dl_train, dl_valid, dl_test, nFactorMax=20):
	### naive and Kelly only
	nFactors_list = np.arange(min(model._individual_feature_dim, nFactorMax)) + 1
	count = len(nFactors_list)
	SR = np.zeros((3, count), dtype=float)
	UV = np.zeros((3, count), dtype=float)
	Alpha = np.zeros((3, count), dtype=float)
	Alpha_weighted = np.zeros((3, count), dtype=float)
	for k in range(count):
		nFactors = nFactors_list[k]
		model.loadSavedModel(nFactors)
		SR[0, k], UV[0, k], Alpha[0, k], Alpha_weighted[0, k] = model.calculateStatistics(dl_train)
		SR[1, k], UV[1, k], Alpha[1, k], Alpha_weighted[1, k] = model.calculateStatistics(dl_valid)
		SR[2, k], UV[2, k], Alpha[2, k], Alpha_weighted[2, k] = model.calculateStatistics(dl_test)
	return SR, UV, Alpha, Alpha_weighted

def plotStatistics(nFactors, SR, UV, Alpha, Alpha_weighted, plotPath=None, figsize=(8,6)):
	plt.figure('SR', figsize=figsize)
	plt.plot(nFactors, SR[0], label='train')
	plt.plot(nFactors, SR[1], label='valid')
	plt.plot(nFactors, SR[2], label='test')
	plt.ylim(0.0,3.0)
	plt.xlabel('Number of Factors')
	plt.ylabel('Sharpe Ratio')
	plt.legend()
	plt.title('Sharpe Ratio with Numbers of Factors')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'SR.pdf'))
		plt.savefig(os.path.join(plotPath, 'SR.png'))

	plt.figure('UV', figsize=figsize)
	plt.plot(nFactors, UV[0], label='train')
	plt.plot(nFactors, UV[1], label='valid')
	plt.plot(nFactors, UV[2], label='test')
	plt.ylim(0.5,1.0)
	plt.xlabel('Number of Factors')
	plt.ylabel('Unexplained Variation')
	plt.legend()
	plt.title('Unexplained Variation with Numbers of Factors')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'UV.pdf'))
		plt.savefig(os.path.join(plotPath, 'UV.png'))

	plt.figure('Alpha', figsize=figsize)
	plt.plot(nFactors, Alpha[0], label='train')
	plt.plot(nFactors, Alpha[1], label='valid')
	plt.plot(nFactors, Alpha[2], label='test')
	plt.ylim(0.001,0.005)
	plt.xlabel('Number of Factors')
	plt.ylabel('Fama-McBeth Type Alpha')
	plt.legend()
	plt.title('Fama-McBeth Type Alpha with Numbers of Factors')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'Alpha.pdf'))
		plt.savefig(os.path.join(plotPath, 'Alpha.png'))

	plt.figure('Alpha (Weighted)', figsize=figsize)
	plt.plot(nFactors, Alpha_weighted[0], label='train')
	plt.plot(nFactors, Alpha_weighted[1], label='valid')
	plt.plot(nFactors, Alpha_weighted[2], label='test')
	plt.ylim(0.0,0.0006)
	plt.xlabel('Number of Factors')
	plt.ylabel('Fama-McBeth Type Alpha (Weighted)')
	plt.legend()
	plt.title('Fama-McBeth Type Alpha (Weighted) with Numbers of Factors')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'Alpha_weighted.pdf'))
		plt.savefig(os.path.join(plotPath, 'Alpha_weighted.png'))

	plt.show()