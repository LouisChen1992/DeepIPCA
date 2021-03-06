import json
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import deco_print
from src.data.data_layer import DataInRamInputLayer
from src.model.model_IPCA_GDFFN import ModelIPCA_GDFFN

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')
tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_integer('nFactor', 1, 'Number of Factors')
tf.flags.DEFINE_boolean('isTrain', True, 'True if train, False if evaluate')

FLAGS = tf.flags.FLAGS

def main(_):
	with open(FLAGS.config, 'r') as file:
		config = json.load(file)
	deco_print('Read the following in config: ')
	print(json.dumps(config, indent=4))

	deco_print('Creating data layer')
	dl_train = DataInRamInputLayer(config['individual_feature_file'])
	dl_valid = DataInRamInputLayer(config['individual_feature_file_valid'])
	dl_test = DataInRamInputLayer(config['individual_feature_file_test'])
	deco_print('Data layer created')

	model = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_train'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=FLAGS.nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=FLAGS.logdir, 
							dl=dl_train, 
							is_train=True)
	model_valid = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_valid'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=FLAGS.nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=FLAGS.logdir, 
							dl=dl_valid, 
							is_train=False, 
							force_var_reuse=True)
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	sess = tf.Session(config=sess_config)

	if FLAGS.isTrain:
		model.randomInitialization(sess)
		model.train(sess, model_valid, numEpoch=config['num_epoch'], subEpoch=config['sub_epoch'])

	model_test = ModelIPCA_GDFFN(individual_feature_dim=config['individual_feature_dim'], 
							tSize=config['tSize_test'], 
							hidden_dims=config['hidden_dims'], 
							nFactor=FLAGS.nFactor, 
							lr=config['lr'], 
							dropout=config['dropout'],
							logdir=FLAGS.logdir, 
							dl=dl_test, 
							is_train=False, 
							force_var_reuse=True)
	model.loadSavedModel(sess)
	w = model.getMarkowitzWeight(sess)
	stats = pd.DataFrame(np.zeros((4,3)), columns=['train', 'valid', 'test'], index=['SR', 'UV', 'Alpha', 'Alpha_weighted'])
	stats.loc[:,'train'] = model.calculateStatistics(sess, w)
	stats.loc[:,'valid'] = model_valid.calculateStatistics(sess, w)
	stats.loc[:,'test'] = model_test.calculateStatistics(sess, w)
	print(stats)

if __name__ == '__main__':
	tf.app.run()