import argparse
import os
import json

class HyperParameterSpace:
	def __init__(self):
		"""Define Hyper-Parameters. 

		Returns:
			hp: Map from hyper-parameter to a list of choices. 
		"""
		self.hp = {}
		self.hp['hidden_dims'] = [[32], [16], [8], [4],
								[32,32], [16,16], [8,8], [4,4],
								[32,16], [16,8], [8,4], [4,2]]
		self.hp['lr'] = [0.1, 0.01, 0.001]
		self.hp['dropout'] = [0.99, 0.95, 0.9]
		### ADD CODE HERE

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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Create configurations')
	parser.add_argument('--src', help='Source config file')
	parser.add_argument('--tgt', help='Path to save config files')
	args = parser.parse_args()
	with open(args.src, 'r') as file:
		config = json.load(file)
	hp = HyperParameterSpace()
	for config_idx, (hp_idx, item) in enumerate(hp.iterateAllCombinations()):
		for param, val in item:
			config[param] = val
		file = 'config_%d_hp_%s.json' %(config_idx, hp_idx)
		with open(os.path.join(args.tgt, file), 'w') as file:
			json.dump(config, file)
