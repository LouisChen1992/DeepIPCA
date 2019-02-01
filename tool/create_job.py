import argparse
import os

if __name__ == '__main__':
	root_dir = '..'
	main_code = 'run_FFN.py'
	config_path = os.path.join(root_dir, 'configs/IPCA_FFN')
	logdir_path = os.path.join(root_dir, 'model/IPCA_FFN')
	version = 'v1'
	notification = False
	sep = '='
	options = ['--config', '--logdir', '--nFactor']
	nFactor_list = [i+1 for i in range(20)]

	config_list = [item for item in os.listdir(config_path) if item.endswith('.json')]
	config_count = len(config_list)

	for job_id in range(config_count):
		config_file = config_list[job_id]
		job_path = os.path.join(root_dir, 'job_%d.sh' %job_id)
		with open(job_path, 'w') as file:
			for nFactor in nFactor_list:
				values = [os.path.join(config_path, config_file), os.path.join(logdir_path, version), str(nFactor)]
				cmd = ' '.join(['python3', main_code] + [sep.join([option, value]) for option, value in zip(options, values)])
				file.write(cmd+'\n')
			if notification:
				cmd = 'python3 ~/tool/sendEmail.py --FROM lychSendOnly@hotmail.com --TO lych@stanford.edu --p send1Email --s Job\ %d\ Finished! --MSG job_path' %job_id
				file.write(cmd+'\n')