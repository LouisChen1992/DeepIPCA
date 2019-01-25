import numpy as np

from src.utils import deco_print

class DataInRamInputLayer:
	""" Class that defines a data input layer. 
	"""
	def __init__(self, 
				pathIndividualFeature, 
				pathMacroFeature=None,
				meanMacroFeature=None, 
				stdMacroFeature=None):
		"""Initialize a data input layer. 

		Arguments: 
			pathIndividualFeature: Path to individual features. 
			pathMacroFeature: Path to macro features. 
			meanMacroFeature: Mean to normalize macro features, calculated if necessary. 
			stdMacroFeature: Std to normalize macro features, calculated if necessary. 
		"""
		self._UNK = -99.99
		self._load_individual_feature(pathIndividualFeature)
		self._load_macro_feature(pathMacroFeature, meanMacroFeature, stdMacroFeature)

	def _create_var_idx_associations(self, varList):
		"""Create variable - index mapping. 

		Arguments: 
			varList: List of variables. 
		Returns:
			idx2var: Map from index to variable. 
			var2idx: Map from variable to index. 
		"""
		idx2var = {idx:var for idx, var in enumerate(varList)}
		var2idx = {var:idx for idx, var in enumerate(varList)}
		return idx2var, var2idx

	def _load_individual_feature(self, pathIndividualFeature):
		"""Load individual features. 

		Arguments: 
			pathIndividualFeature: Path to individual features. 
		"""
		tmp = np.load(pathIndividualFeature)
		data = tmp['data']
		
		self._return = data[:,:,0]
		self._individualFeature = data[:,:,1:]
		self._mask = (self._return != self._UNK)

		self._idx2date, self._date2idx = self._create_var_idx_associations(tmp['date'])
		self._idx2permno, self._permno2idx = self._create_var_idx_associations(tmp['permno'])
		self._idx2var, self._var2idx = self._create_var_idx_associations(tmp['variable'][1:])
		self._dateCount, self._permnoCount, self._varCount = data.shape
		self._varCount -= 1

	def _load_macro_feature(self, pathMacroFeature, meanMacroFeature=None, stdMacroFeature=None):
		"""Load macro features, normalized by mean and std. 

		Arguments: 
			pathMacroFeature: Path to macro features. 
			meanMacroFeature: Mean to normalize macro features, calculated if necessary. 
			stdMacroFeature: Std to normalize macro features, calculated if necessary. 
		"""
		if pathMacroFeature is None:
			self._macroFeature = np.empty(shape=[self._dateCount, 0])
			self._meanMacroFeature = None
			self._stdMacroFeature = None
		else:
			tmp = np.load(pathMacroFeature)
			self._macroFeature = tmp['data']
			if meanMacroFeature is None or stdMacroFeature is None:
				self._meanMacroFeature = self._macroFeature.mean(axis=0)
				self._stdMacroFeature = self._macroFeature.std(axis=0)
			else:
				self._meanMacroFeature = meanMacroFeature
				self._stdMacroFeature = stdMacroFeature
			self._macroFeature -= self._meanMacroFeature
			self._macroFeature /= self._stdMacroFeature

			self._idx2var_macro, self._var2idx_macro = self._create_var_idx_associations(tmp['variable'][macro_idx])
			self._varCount_macro = self._macroFeature.shape[1]

	def getDateCountList(self):
		"""Get valid date counts for all permnos. 
		"""
		return np.sum(self._mask, axis=0)

	def getDateList(self):
		"""Get all dates. 
		"""
		return [self._idx2date[i] for i in range(self._dateCount)]

	def getPermnoList(self):
		"""Get all permnos. 
		"""
		return [self._idx2permno[i] for i in range(self._permnoCount)]

	def getIndividualFeatureList(self):
		"""Get all individual features. 
		"""
		return [self._idx2var[i] for i in range(self._varCount)]

	def getDateByIdx(self, idx):
		"""Get date by its index. 

		Arguments: 
			idx: Index in [0, self._dateCount). 
		"""
		return self._idx2date[idx]

	def getPermnoByIdx(self, idx):
		"""Get permno by its index. 

		Arguments: 
			idx: Index in [0, self._permnoCount). 
		"""
		return self._idx2permno[idx]

	def getIndividualFeatureByIdx(self, idx):
		"""Get individual feature by its index. 

		Arguments: 
			idx: Index in [0, self._varCount). 
		"""
		return self._idx2var[idx]

	def getIdxByIndividualFeature(self, var):
		"""Get index by individual feature. 

		Arguments: 
			var: Individual feature. 
		"""
		return self._var2idx[var]

	def getMacroFeatureList(self):
		"""Get all macro features. 
		"""
		return [self._idx2var_macro[i] for i in range(self._varCount_macro)]

	def getMacroFeatureByIdx(self, idx):
		"""Get macro feature by its index. 

		Arguments: 
			idx: Index in [0, self._varCount_macro). 
		"""
		return self._idx2var_macro[idx]

	def getFeatureByIdx(self, idx):
		"""Get individual or macro feature by its (global) index. 

		Arguments: 
			idx: Index in [0, self._varCount + self._varCount_macro). 
		"""
		if idx < self._varCount:
			return self.getIndividualFeatureByIdx(idx)
		else:
			return self.getMacroFeatureByIdx(idx - self._varCount)

	def getMacroFeatureMeanStd(self):
		"""Get mean and std of macro features. 
		"""
		return self._meanMacroFeature, self._stdMacroFeature

	def iterateOneEpoch(self, subEpoch=False):
		"""Go through the whole data input layer. 

		Arguments: 
			subEpoch: Number of times to go through the whole data input layer. 
		"""
		if subEpoch:
			for _ in range(subEpoch):
				yield self._macroFeature, self._individualFeature, self._return, self._mask
		else:
			yield self._macroFeature, self._individualFeature, self._return, self._mask