import argparse
import os
import json

from src.hyper_parameter.HyperParameterSpace import HyperParameterSpace

parser = argparse.ArgumentParser(description='Create configurations')
parser.add_argument('--src', help='Source config file')
parser.add_argument('--tgt', help='Path to save config files')
args = parser.parse_args()
with open(args.src, 'r') as file:
	config = json.load(file)

hp_dict = {}
hp_dict['hidden_dims'] = [[32], [16], [8], [4],
						[32,32], [16,16], [8,8], [4,4],
						[32,16], [16,8], [8,4], [4,2]]
hp_dict['lr'] = [0.1, 0.01, 0.001]
hp_dict['dropout'] = [0.99, 0.95, 0.9]

hp = HyperParameterSpace(hp_dict)
for config_idx, (hp_idx, item) in enumerate(hp.iterateAllCombinations()):
	for param, val in item:
		config[param] = val
	file = 'config_%d_hp_%s.json' %(config_idx, hp_idx)
	with open(os.path.join(args.tgt, file), 'w') as file:
		json.dump(config, file)