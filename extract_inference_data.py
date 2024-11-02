import config, ee_dnn
import argparse, logging, os, torch
from tqdm import tqdm
import numpy as np
import torch.optim as optim
import torch.nn as nn
import pandas as pd
#import utils

def extracting_ee_inference_data(args, test_loader, model, device, distortion_level):

	n_exits = args.n_branches + 1	
	conf_list, correct_list, delta_inf_time_list, cum_inf_time_list = [], [], [], []
	prediction_list, target_list = [], []

	model.eval()
	with torch.no_grad():
		for (data, target) in tqdm(test_loader):	

			# Convert data and target into the current device.
			data, target = data.to(device), target.to(device)

			# Obtain confs and predictions for each side branch.
			_, conf_branches, predictions_branches, delta_inf_time_branches, cum_inf_time_branches = model.forwardTraining(data)

			conf_list.append([conf_branch.item() for conf_branch in conf_branches]), delta_inf_time_list.append(delta_inf_time_branches)
			cum_inf_time_list.append(cum_inf_time_branches)

			correct_list.append([predictions_branches[i].eq(target.view_as(predictions_branches[i])).sum().item() for i in range(n_exits)])
			target_list.append(target.item()), prediction_list.append([predictions_branches[i].item() for i in range(n_exits)])

	conf_list, correct_list, delta_inf_time_list = np.array(conf_list), np.array(correct_list), np.array(delta_inf_time_list)
	cum_inf_time_list, prediction_list = np.array(cum_inf_time_list), np.array(prediction_list)

	#accuracy_branches = [sum(correct_list[:, i])/len(correct_list[:, i]) for i in range(n_exits)]

	#print("Accuracy: %s"%(accuracy_branches))
	result_dict = {"distortion_type": len(target_list)*[args.distortion_type], 
	"distortion_level": len(target_list)*[distortion_level],
	"device": len(target_list)*[str(device)],
	"target": target_list}

	for i in range(n_exits):
		result_dict["conf_branch_%s"%(i+1)] = conf_list[:, i]
		result_dict["correct_branch_%s"%(i+1)] = correct_list[:, i]
		result_dict["delta_inf_time_branch_%s"%(i+1)] = delta_inf_time_list[:, i]
		result_dict["cum_inf_time_branch_%s"%(i+1)] = cum_inf_time_list[:, i]
		result_dict["prediction_branch_%s"%(i+1)] = prediction_list[:, i]

	#Converts to a DataFrame Format.
	df = pd.DataFrame(np.array(list(result_dict.values())).T, columns=list(result_dict.keys()))

	# Returns confidences and predictions into a DataFrame.
	return df

def main(args):

	n_classes = config.n_class_dict[args.dataset_name]

	print(args.use_gpu)
	sys.exit()

	device = torch.device('cuda' if (torch.cuda.is_available() and args.use_gpu) else 'cpu')

	models_dir_path = os.path.join(config.DIR_PATH, args.model_name, "models")

	model_path = os.path.join(models_dir_path, "ee_model_%s_%s_branches_%s_id_%s.pth"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))

	indices_path = os.path.join(config.DIR_PATH, "indices_%s.pt"%(args.dataset_name))

	inf_data_dir_path = os.path.join(config.DIR_PATH, args.model_name, "inf_data")
	os.makedirs(inf_data_dir_path, exist_ok=True)

	inf_data_path = os.path.join(inf_data_dir_path, "inf_data_ee_%s_%s_branches_%s_id_%s.csv"%(args.model_name, 
		args.n_branches, args.loss_weights_type, args.model_id))
	
	ee_model = utils.load_eednn_model(args, n_classes, model_path, device)

	distortion_level_list = config.distortion_level_dict[args.distortion_type]

	for distortion_level in distortion_level_list:
		print("Distortion Type: %s, Distortion Level: %s"%(args.distortion_type, distortion_level))

		dist_dataset_path = os.path.join(config.distorted_dataset_path, args.distortion_type, str(distortion_level))

		_, _, test_loader = utils.load_caltech256(args, dist_dataset_path, indices_path)

		df_inf_data = extracting_ee_inference_data(args, test_loader, ee_model, device, distortion_level)

		df_inf_data.to_csv(inf_data_path, mode='a', header=not os.path.exists(inf_data_path))


if (__name__ == "__main__"):
	# Input Arguments to configure the early-exit model .
	parser = argparse.ArgumentParser(description="Extract the confidences obtained by DNN inference for next experiments.")

	#We here insert the argument dataset_name. 
	#The initial idea is this novel calibration method evaluates three dataset for image classification: cifar10, cifar100 and
	#caltech256. First, we implement caltech256 dataset.
	parser.add_argument('--dataset_name', type=str, default=config.dataset_name, 
		choices=["caltech256", "cifar10"], help='Dataset name.')

	#We here insert the argument model_name. 
	#We evalue our novel calibration method Offloading-driven Temperature Scaling in four early-exit DNN:
	#MobileNet
	parser.add_argument('--model_name', type=str, default=config.model_name, choices=["mobilenet", "alexnet"], 
		help='DNN model name (default: %s)'%(config.model_name))

	parser.add_argument('--input_dim', type=int, default=330, help='Input Dim.')

	parser.add_argument('--dim', type=int, default=300, help='Image dimension')

	parser.add_argument('--seed', type=int, default=config.seed, help='Seed.')

	parser.add_argument('--use_gpu', type=bool, default=config.use_gpu, help='Use GPU? Default: %s'%(config.use_gpu))

	parser.add_argument('--n_branches', type=int, help='Number of side branches.')

	parser.add_argument('--exit_type', type=str, default=config.exit_type, 
		help='Exit Type. Default: %s'%(config.exit_type))

	parser.add_argument('--distribution', type=str, default=config.distribution, 
		help='Distribution of the early exits. Default: %s'%(config.distribution))

	parser.add_argument('--pretrained', type=bool, default=config.pretrained, help='Backbone DNN is pretrained.')

	parser.add_argument('--model_id', type=int, help='Model_id.')

	parser.add_argument('--loss_weights_type', type=str, help='loss_weights_type.')

	#This argument defines the ratio to split the Traning Set, Val Set, and Test Set.
	parser.add_argument('--split_ratio', type=float, default=config.split_ratio, help='Split Ratio')

	args = parser.parse_args()

	main(args)