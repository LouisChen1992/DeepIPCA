import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.data_layer import DataInRamInputLayer
from src.model.model_IPCA_FFN import ModelIPCA_FFN
from src.model.model_IPCA_GDFFN import ModelIPCA_GDFFN
from src.hyper_parameter.HyperParameterSpace import HyperParameterSpace

tf.flags.DEFINE_string('model', '', 'Model')
tf.flags.DEFINE_string('config_path', '', 'Path to config files')
tf.flags.DEFINE_string('logdir_path', '', 'Path to logdir')
FLAGS = tf.flags.FLAGS

def test(config, logdir, nFactor, IPCA_model):
	dl_train = DataInRamInputLayer(config['individual_feature_file'])
	dl_valid = DataInRamInputLayer(config['individual_feature_file_valid'])
	dl_test = DataInRamInputLayer(config['individual_feature_file_test'])

	tf.reset_default_graph()
	if IPCA_model == 'FFN':
		model, model_valid, model_test = construct_FFN(config, logdir, nFactor, dl_train, dl_valid, dl_test)
	elif IPCA_model == 'GDFFN':
		model, model_valid, model_test = construct_FFN(config, logdir, nFactor, dl_train, dl_valid, dl_test)

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

def construct_FFN(config, logdir, nFactor, dl_train, dl_valid, dl_test):
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
	return model, model_valid, model_test

def construct_GDFFN(config, logdir, nFactor, dl_train, dl_valid, dl_test):
	model = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_train'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=logdir, 
							dl=dl_train, 
							is_train=False)
	model_valid = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_valid'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=logdir, 
							dl=dl_valid, 
							is_train=False, 
							force_var_reuse=True)
	model_test = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_test'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=logdir, 
							dl=dl_test, 
							is_train=False, 
							force_var_reuse=True)
	return model, model_valid, model_test

def main(_):
	hp_dict = {}
	if FLAGS.model == 'FFN':
		hp_dict['hidden_dims'] = [[32], [16], [8], [4],
							[32,32], [16,16], [8,8], [4,4],
							[32,16], [16,8], [8,4], [4,2]]
		hp_dict['lr'] = [0.002, 0.001, 0.0005, 0.0002]
		hp_dict['dropout'] = [0.99, 0.95, 0.9]
	elif FLAGS.model == 'GDFFN':
		hp_dict['hidden_dims'] = [[64], [32], [16], 
							[64,64], [32,32], [16,16],
							[64,32], [32,16]]
		hp_dict['lr'] = [0.002, 0.001, 0.0005, 0.0002]
		hp_dict['dropout'] = [0.99, 0.95, 0.9]
	else:
		raise ValueError('Invalid Model! ')

	hp = HyperParameterSpace(hp_dict)
	params = hp.getParamsName()

	config_list = sorted([item for item in os.listdir(FLAGS.config_path) if item.endswith('.json')])
	config_count = len(config_list)
	df_list = {'idx':np.arange(config_count)}
	for param in params:
		df_list[param] = np.empty(shape=(config_count,), dtype=str)
	df = pd.DataFrame(df_list)
	df.set_index('idx', inplace=True)
	df_map = {i+1:df.copy() for i in range(20)}

	for i in range(config_count):
		config_file = config_list[i]
		path_config = os.path.join(FLAGS.config_path, config_file)
		logdir = os.path.join(FLAGS.logdir_path, config_file.rstrip('.json'))
		
		hp_idx = [int(item) for item in config_file.rstrip('.json').split('_', 3)[-1].split('_')]
		hp_val = hp.idx2Val(hp_idx)

		with open(path_config, 'r') as file:
			config = json.load(file)
		nFactor_list = sorted([int(item) for item in os.listdir(logdir)])
		for nFactor in nFactor_list:
			stats = test(config, logdir, nFactor, FLAGS.model)
			for param, val in zip(params, hp_val):
				df_map[nFactor].loc[i, param] = str(val)
			df_map[nFactor].loc[i, 'SR_train'] = stats.values[0,0]
			df_map[nFactor].loc[i, 'SR_valid'] = stats.values[0,1]
			df_map[nFactor].loc[i, 'SR_test'] = stats.values[0,2]
	store = pd.HDFStore(os.path.join(FLAGS.logdir_path, 'summary.h5'))
	for i in range(20):
		store['nFactor_%d' %(i+1)] = df_map[i+1]
	store.close()

if __name__ == '__main__':
	tf.app.run()