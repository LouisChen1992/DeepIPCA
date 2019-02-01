import os
import json
import pandas as pd
import tensorflow as tf

from src.data.data_layer import DataInRamInputLayer
from src.model.model_IPCA_FFN import ModelIPCA_FFN
from src.hyper_parameter.HyperParameterSpace import HyperParameterSpace

tf.flags.DEFINE_string('config_path', '', 'Path to config files')
tf.flags.DEFINE_string('logdir_path', '', 'Path to logdir')
FLAGS = tf.flags.FLAGS

def test(config, logdir, nFactor):
	dl_train = DataInRamInputLayer(config['individual_feature_file'])
	dl_valid = DataInRamInputLayer(config['individual_feature_file_valid'])
	dl_test = DataInRamInputLayer(config['individual_feature_file_test'])
	
	tf.reset_default_graph()
	model = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
						tSize=config['tSize_train'], 
						hidden_dims=config['hidden_dims'], 
						nFactor=nFactor, 
						lr=config['lr'], 
						dropout=config['dropout'],
						logdir=logdir, 
						dl=dl_train, 
						is_train=False)
	model_valid = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_valid'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=logdir, 
							dl=dl_valid, 
							is_train=False, 
							force_var_reuse=True)
	model_test = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_test'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=logdir, 
							dl=dl_test, 
							is_train=False, 
							force_var_reuse=True)
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	sess = tf.Session(config=sess_config)
	model.loadSavedModel(sess)
	w = model.getMarkowitzWeight(sess)
	stats = pd.DataFrame(np.zeros((4,3)), columns=['train', 'valid', 'test'], index=['SR', 'UV', 'Alpha', 'Alpha_weighted'])
	stats.loc[:,'train'] = model.calculateStatistics(sess, w)
	stats.loc[:,'valid'] = model_valid.calculateStatistics(sess, w)
	stats.loc[:,'test'] = model_test.calculateStatistics(sess, w)
	return stats

def main(_):
	hp = HyperParameterSpace()
	params = hp.getParamsName()
	param_types = hp.getParamsType()
	config_list = sorted([item for item in os.listdir(FLAGS.config_path) if item.endswith('.json')])
	config_count = len(config_list)
	df_list = {'idx':np.arange(config_count)}
	for param, param_type in zip(params, param_types):
		df_list[param] = np.empty(shape=(config_count,), dtype=param_type)
	df = pd.DataFrame(df_list)
	df.set_index('idx', inplace=True)

	for i in range(config_count):
		config_file = config_list[i]
		path_config = os.path.join(FLAGS.config_path, config_file)
		logdir = os.path.join(FLAGS.logdir_path, config_file.rstrip('.json'))
		
		hp_idx = config_file.rstrip('.json').split('_', 3)
		hp_val = hp.idx2Val(hp_idx)

		with open(path_config, 'r') as file:
			config = json.load(file)
		nFactor_list = sorted([int(item) for item in os.listdir(logdir)])
		for nFactor in nFactor_list:
			stats = test(config, logdir, nFactor)
			for param, val in zip(params, hp_val):
				df.loc[i, param] = val
			df.loc[i, 'nFactor'] = nFactor
			df.loc[i, ['SR_train','SR_valid','SR_test']] = stats[0].values
	print(df)

if __name__ == '__main__':
	tf.app.run()