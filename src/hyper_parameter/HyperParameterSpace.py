class HyperParameterSpace:
	def __init__(self, hp):
		"""Define Hyper-Parameters. 
		"""
		self.hp = hp
		self.params = sorted(list(self.hp.keys()))

	def iterateAllCombinations(self):
		"""Go through the whole combinations of hyper-parameters. 
		"""
		idx = [0] * len(self.params)
		maxIdx = [len(self.hp[param]) for param in self.params]
		while True:
			yield self.idx2Str(idx), [(param, self.hp[param][i]) for param, i in zip(self.params, idx)]
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
		return [self.hp[param][i] for param, i in zip(self.params, idx)]

	def getParamsName(self):
		return list(self.hp.keys())

	def getParamsType(self):
		return list([type(self.hp[param][0]) for param in self.params])