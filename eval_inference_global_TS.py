import config, ee_dnns, sys, utils, ts
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd


def main(args):

	n_classes = 257

	device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

	device = torch.device(device_str)

	model_path = os.path.join(config.DIR_PATH, args.model_name, "models", "ee_model_%s_%s_branches_%s_id_%s.pth"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "global_TS_inf_data_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))
	
	ee_model = ee_dnns.load_eednn_model(args, n_classes, model_path, device)

	dataset_path = os.path.join("datasets", args.dataset_name)

	_, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path)


	threshold_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
	temp_init = 1.0
	max_iter = 1000

	for threshold in threshold_list:
		print("Threshold: %s"%(threshold))
		global_ts_calib_model = ts.GlobalTemperatureScaling(ee_model, device, temp_init, max_iter, args.n_branches, threshold)
		global_ts_calib_model.run(val_loader)





if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech-256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=300, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--use_gpu', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, default=1, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	parser.add_argument('--model_id', type=int, default=3, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, default="decrescent", help='loss_weights_type.')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	parser.add_argument('--batch_size_train', type=int, default=config.batch_size_train, 
		help='Train Batch Size. Default: %s'%(config.batch_size_train))

	parser.add_argument('--location', type=str, help='Which machine extracts the inference data', choices=["pechincha", "jetson", "RO"])

	args = parser.parse_args()

	main(args)