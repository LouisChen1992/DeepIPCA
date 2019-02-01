class HyperParameterSpace:
	def __init__(self):
		"""Define Hyper-Parameters. 

		Returns:
			hp: Map from hyper-parameter to a list of choices. 
		"""
		self.hp = {}

		### ADD CODE HERE
		self.hp['hidden_dims'] = [[32], [16], [8], [4],
								[32,32], [16,16], [8,8], [4,4],
								[32,16], [16,8], [8,4], [4,2]]
		self.hp['lr'] = [0.1, 0.01, 0.001]
		self.hp['dropout'] = [0.99, 0.95, 0.9]

	def iterateAllCombinations(self):
		"""Go through the whole combinations of hyper-parameters. 
		"""
		params = list(self.hp.keys())
		idx = [0] * len(params)
		maxIdx = [len(self.hp[param]) for param in params]
		while True:
			yield self.idx2Str(idx), [(param, self.hp[param][i]) for param, i in zip(params, idx)]
			self.addOneIdx(idx, maxIdx)
			if self.isZeroIdx(idx):
				break

	def isZeroIdx(self, idx):
		"""Whether idx is zero. 
		"""
		for i in range(len(idx)):
			if idx[i] > 0:
				return False
		return True

	def addOneIdx(self, idx, maxIdx):
		"""Add idx by one. 
		"""
		assert(len(idx)==len(maxIdx))
		i = len(idx) - 1
		while i >= 0:
			idx[i] += 1
			if idx[i] == maxIdx[i]:
				idx[i] = 0
				i -= 1
			else:
				return

	def idx2Str(self, idx):
		"""Convert idx to string. 
		"""
		return '_'.join([str(num) for num in idx])

	def idx2Val(self, idx):
		"""Convert idx to values. 
		"""
		params = list(self.hp.keys())
		return [self.hp[param][i] for param, i in zip(params, idx)]

	def getParamsName(self):
		return list(self.hp.keys())

	def getParamsType(self):
		params = list(self.hp.keys())
		return list([type(self.hp[param][0]) for param in params])