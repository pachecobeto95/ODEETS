import os, time, sys, json, os, argparse, config
#import config, utils, temperature_scaling, ee_nn
#from early_exit_dnn import Early_Exit_DNN
import numpy as np
import pandas as pd
#import spsa3 as spsa


def main(args):


	n_classes = 257

	input_dim, dim = 330, 300





	inf_data_cloud_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_local_server.csv"%(args.model_name, args.n_branches, args.model_id))
	#inf_data_cloud_path = os.path.join(config.DIR_NAME, "inference_data", "inference_data_%s_%s_branches_%s.csv"%(args.model_name, args.n_branches, args.model_id))
	#val_inf_data_path = os.path.join(config.DIR_NAME, "new_inference_data", "val_inference_data_%s_%s_branches_%s.csv"%(args.model_name, args.n_branches, args.model_id))
	inf_data_device_path = os.path.join(config.DIR_NAME, "new_inference_data", "inference_data_%s_%s_branches_%s_jetson_nano.csv"%(args.model_name, args.n_branches, args.model_id))

	#resultsPath = os.path.join(config.DIR_NAME, "exp_beta_analysis_%s_%s_branches_with_overhead_%s.csv"%(args.model_name, args.n_branches, args.overhead))
	resultsPath = os.path.join(config.DIR_NAME, "test_theo_result_overhead_%s"%(args.overhead))
	#resultsPath = os.path.join(config.DIR_NAME, "theo_concorrents_beta_analysis_%s_%s_branches_overhead_%s_2.csv"%(args.model_name, args.n_branches, args.overhead))

	global_ts_path = os.path.join(config.DIR_NAME, "alternative_temperature_%s_%s_branches_id_%s_rodrigo_version_2.csv"%(args.model_name, args.n_branches, args.model_id))

	threshold_list = [0.8]
	beta_list = np.arange(0, 205, 5)
	beta_list = [10000]


	df_inf_data_cloud = pd.read_csv(inf_data_cloud_path)
	df_inf_data_device = pd.read_csv(inf_data_device_path)
	#overhead_list = [5, 10, 15]
	overhead_list = [args.overhead]

	for overhead in overhead_list:
		#for n_branches_edge in reversed(range(1, args.n_branches+1)):
		for n_branches_edge in [args.n_branches]:

			for threshold in threshold_list:
				print("Overhead: %s, Nr Branches: %s, Threshold: %s"%(overhead, n_branches_edge, threshold))

				run_theoretical_beta_analysis(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, n_branches_edge, 
					beta_list, resultsPath, overhead, mode, calib_mode="beta_calib")			

				#runNoCalibInference(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, n_branches_edge, 
				#	resultsPath, overhead, calib_mode="no_calib")

				#runGlobalTemperatureScalingInference(args, df_inf_data_cloud, df_inf_data_cloud, df_inf_data_device, threshold, n_branches_edge, 
				#	resultsPath, global_ts_path, overhead, calib_mode="global_TS")


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Learning the Temperature driven for offloading.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256"], help='Dataset name (default: Caltech-256)')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet, ResNet18, ResNet152, VGG16
	parser.add_argument('--model_name', type=str, choices=["mobilenet", "resnet18", "resnet152", "vgg16"], 
		help='DNN model name (default: mobilenet)')

	# This argument defines the backbone DNN is pretrained.
	parser.add_argument('--pretrained', type=bool, default=config.pretrained, 
		help='Is backbone DNN pretrained? Default: %s'%(config.pretrained))

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution. Default: %s'%(config.distribution))

	parser.add_argument('--n_branches', type=int, default=config.n_branches, 
		help='Number of side branches. Default: %s'%(config.n_branches))

	parser.add_argument('--max_iter', type=int, default=config.max_iter, 
		help='Number of epochs. Default: %s'%(config.max_iter))

	parser.add_argument('--a0', type=int, default=config.a0, 
		help='a0. Default: %s'%(config.a0))

	parser.add_argument('--c', type=int, default=config.c, 
		help='c. Default: %s'%(config.c))

	parser.add_argument('--alpha', type=int, default=config.alpha, 
		help='alpha. Default: %s'%(config.alpha))

	parser.add_argument('--gamma', type=int, default=config.gamma, 
		help='gamma. Default: %s'%(config.gamma))

	parser.add_argument('--step', type=float, default=config.step, 
		help="Step of beta. Default: %s"%(config.step))	

	parser.add_argument('--model_id', type=int, default=1)	

	parser.add_argument('--input_dim', type=int, default=330)

	parser.add_argument('--dim', type=int, default=300, help='Dim. Default: %s')

	args = parser.parse_args()

	main(args)
