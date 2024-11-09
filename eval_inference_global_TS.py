import config, ee_dnns, sys, utils, ts
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd

def run_ee_dnn_inference(args, model, inf_time_data, test_loader, threshold, overhead, device):

	correct_list, inf_time_list, isOffloaded_list, exit_branch_list, correct_edge_list = [], [], [], [], []

	model.eval()
	
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# Obtain confs and predictions for each side branch.
			prediction, isOffloaded, exit_branch = model.forwardGlobalTSInference(data, threshold)

			correct = prediction.eq(target.view_as(prediction)).sum().item()
			inf_time = compute_inference_time(inf_time_data, exit_branch, args.n_branches, overhead)
			

			inf_time_list.append(inf_time), isOffloaded_list.append(isOffloaded), correct_list.append(correct), 

			if(isOffloaded):
				correct_edge_list.append(correct)


	correct_list, inf_time_list, isOffloaded_list = np.array(correct_list), np.array(inf_time_list), np.array(isOffloaded_list)


	overall_accuracy = sum(correct_list)/len(correct_list)
	accuracy_edge = sum(correct_edge_list)/len(correct_edge_list)
	offloading_prob = sum(isOffloaded_list)/len(isOffloaded_list)

	
	result_dict = {"accuracy_edge": accuracy_edge, "offloading_prob": offloading_prob, "overall_accuracy": overall_accuracy, 
	"avg_inf_time": , "std_inf_time": , "threshold": threshold, "overhead": overhead}

	temp_data = model.get_temperature_data()

	result_dict.update(temp_data)


	sys.exit()


	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df


def get_inference_time(df_cloud, df_edge):

	print(df_cloud)

	print(df_edge)

	sys.exit()


def main(args):

	n_classes = 257

	device_str = 'cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu'

	device = torch.device(device_str)

	model_path = os.path.join(config.DIR_PATH, args.model_name, "models", "ee_model_%s_%s_branches_%s_id_%s.pth"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inference_data")
	
	inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))

	edge_inf_data_path = os.path.join(inf_data_dir_path, "inference_data_mobilenet_1_branches_1_jetson_nano.csv")

	result_path = os.path.join(config.DIR_PATH, args.model_name, "results", "globalTS")
	os.makedirs(result_path, exist_ok=True)
	
	ee_model = ee_dnns.load_eednn_model(args, n_classes, model_path, device)

	dataset_path = os.path.join("datasets", args.dataset_name)

	_, val_loader, test_loader = utils.load_caltech256(args, dataset_path, indices_path)

	df_cloud, df_edge = pd.read_csv(inf_data_path), pd.read_csv(edge_inf_data_path)

	inf_time_cloud, inf_time_edge = get_inference_time(df_cloud, df_edge)

	for threshold in config.threshold_list:
		print("Threshold: %s"%(threshold))
		global_ts_calib_model = ts.GlobalTemperatureScaling(ee_model, device, args.temp_init, args.max_iter, args.n_branches, threshold)
		global_ts_calib_model.run(val_loader)

		for overhead in config.overhead_list:
			inf_result = run_ee_dnn_inference(global_ts_calib_model, test_loader, threshold, overhead, device)





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


	parser.add_argument('--temp_init', type=float, default=1.0, help='Initial temperature to start the Temperature Scaling')

	parser.add_argument('--max_iter', type=float, default=1000, help='Max Interations to optimize the Temperature Scaling')

	args = parser.parse_args()

	main(args)