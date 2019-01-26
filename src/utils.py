import numpy as np

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