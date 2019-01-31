import json
import numpy as np
import pandas as pd
import tensorflow as tf

from src.utils import deco_print
from src.data.data_layer import DataInRamInputLayer
from src.model.model_IPCA_naive import ModelIPCA_Naive
from src.model.model_IPCA_FFN import ModelIPCA_FFN

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')
tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_boolean('randomInitFactors', False, 'Initialize factors randomly')
tf.flags.DEFINE_boolean('evalStats', True, 'Calculate statistics')

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

	model = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
						tSize=config['tSize_train'], 
						hidden_dims=config['hidden_dims'], 
						nFactor=config['nFactor'], 
						lr=config['lr'], 
						dropout=config['dropout'],
						logdir=FLAGS.logdir, 
						dl=dl_train, 
						is_train=True)
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	sess = tf.Session(config=sess_config)
	model.randomInitialization(sess)

	if FLAGS.randomInitFactors:
		initial_F = None
	else:
		model_naive = ModelIPCA_Naive(46, 'model/IPCA_naive')
		model_naive.loadSavedModel(config['nFactor'])
		initial_F = model_naive.getFactors(dl_train)

	loss_epoch_list = model.train(sess, initial_F=initial_F, 
		numEpoch=config['num_epoch'], maxIter=config['max_iter'], tol=config['tol'])

	if FLAGS.evalStats:
		model_valid = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
								tSize=config['tSize_valid'], 
								hidden_dims=config['hidden_dims'], 
								nFactor=config['nFactor'], 
								lr=config['lr'], 
								dropout=config['dropout'],
								logdir=FLAGS.logdir, 
								dl=dl_valid, 
								is_train=False, 
								force_var_reuse=True)
		model_test = ModelIPCA_FFN(individual_feature_dim=config['individual_feature_dim'], 
								tSize=config['tSize_test'], 
								hidden_dims=config['hidden_dims'], 
								nFactor=config['nFactor'], 
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