import torchvision, torch
import os, sys, time, math
from torch.utils.data import random_split
from PIL import Image
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from itertools import product
import pandas as pd
from torch import Tensor
import functools
import torch.nn.functional as F
from tqdm import tqdm


class ECE(nn.Module):
	"""This method computes ECE metric to measure model's miscalibration"""

	def __init__(self, n_bins=15):
		"""
		n_bins (int): number of confidence interval bins
		"""
		super(ECE, self).__init__()
		bin_boundaries = torch.linspace(0, 1, n_bins + 1)
		self.bin_lowers = bin_boundaries[:-1]
		self.bin_uppers = bin_boundaries[1:]

	def forward(self, logits, labels):
		softmaxes = F.softmax(logits, dim=1)
		confidences, predictions = torch.max(softmaxes, 1)
		accuracies = predictions.eq(labels)

		ece = torch.zeros(1, device=logits.device)
		for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
			# Calculated |confidence - accuracy| in each bin
			in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
			prop_in_bin = in_bin.float().mean()
			if (prop_in_bin.item() > 0):
				accuracy_in_bin = accuracies[in_bin].float().mean()
				avg_confidence_in_bin = confidences[in_bin].mean()
				ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

		return ece


class GlobalTemperatureScaling(nn.Module):
	"""This class implements Global Temperature Scaling for early-exit DNNs"""

	def __init__(self, model, device, temp_init, max_iter, n_branches_edge, threshold, lr=0.001):
		super(GlobalTemperatureScaling, self).__init__()
    
		self.model = model #the model to be calibrated
		self.device = device
		self.temperature_overall = nn.Parameter((temp_init*torch.ones(1)).to(self.device)) #initial temperature to be optimized
		self.lr = lr #learning rate
		self.max_iter = max_iter #maximum iteration to the optimization method
		self.n_branches_edge = n_branches_edge
		self.threshold = threshold


	def temperature_scale(self, logits):
		temperature = self.temperature_overall.unsqueeze(1).expand(logits.size(0), logits.size(1))
		return logits / temperature


	def forwardCalibrationInference(self, x, threshold):
		return self.model.forwardGlobalCalibrationInference(x, threshold, self.temperature_overall)


	def forwardGlobalTS(self, x):
		return self.model.forwardGlobalCalibration(x, self.temperature_overall)


	def get_temperature_data(self):
		# This function probably should live outside of this class, but whatever
		# This method sves the learned temperature parameters.


		result = {"threshold": round(self.threshold, 2), "before_nll": self.before_ts_nll, "after_nll": self.after_ts_nll, "before_ece": self.before_ts_ece, 
		"after_ece": self.after_ts_ece, "temperature": self.temperature_overall.item()}

		#df = pd.DataFrame([result])
		#df.to_csv(self.saveTempPath, mode='a', header=not os.path.exists(self.saveTempPath))

		return result

	def run(self, valid_loader):
		"""
		Tune the tempearature of the model (using the validation set).
		We're going to set it to optimize NLL.
		valid_loader (DataLoader): validation set loader
		p_tar: confidence threshold to decide wheter an input should be classified earlier or not.
		"""
        
		nll_criterion = nn.CrossEntropyLoss().to(self.device)
		ece_criterion = ECE().to(self.device)

		# First: collect all the logits and labels for the validation set
		logits_list, labels_list = [], []

		self.model.eval()
		with torch.no_grad():
			#Run inference over samples from validation dataset
			for data, label in tqdm(valid_loader):
				data, label = data.to(self.device), label.to(self.device)  
				#Check the next row to confirm 
				logits, confs, _, _ = self.model.forwardInference(data, self.threshold)

				logits_list.append(logits), labels_list.append(label)

		logits = torch.cat(logits_list).to(self.device)
		labels = torch.cat(labels_list).to(self.device)

		# Calculate NLL and ECE before temperature scaling
		self.before_ts_nll = nll_criterion(logits, labels).item()
		self.before_ts_ece = ece_criterion(logits, labels).item()
		print('Before temperature - NLL: %.3f, ECE: %.3f' % (self.before_ts_nll, self.before_ts_ece))

		# Next: optimize the temperature w.r.t. NLL
		optimizer = optim.LBFGS([self.temperature_overall], lr=self.lr, max_iter=self.max_iter)

		def eval():
			optimizer.zero_grad()
			loss = nll_criterion(self.temperature_scale(logits), labels)
			loss.backward()
			return loss
		optimizer.step(eval)

		# Calculate NLL and ECE after temperature scaling
		self.after_ts_nll = nll_criterion(self.temperature_scale(logits), labels).item()
		self.after_ts_ece = ece_criterion(self.temperature_scale(logits), labels).item()
		print('Optimal temperature: %.3f' % self.temperature_overall.item())
		print('After temperature - NLL: %.3f, ECE: %.3f' % (self.after_ts_nll, self.after_ts_ece))

		#self.save_temperature(p_tar, before_temperature_nll, after_temperature_nll, before_temperature_ece, after_temperature_ece)
		
		#return self.temperature_overall
		return self

