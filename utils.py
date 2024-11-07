import config
from torchvision import datasets, transforms
import torch, os, sys, requests, cv2
import numpy as np

def save_indices(train_idx, val_idx, test_idx, indices_path):

	data_dict = {"train": train_idx, "val": val_idx, "test": test_idx}
	torch.save(data_dict, indices_path)

def get_indices(dataset, split_ratio, indices_path):
	
	if (not os.path.exists(indices_path)):

		nr_samples = len(dataset)

		indices = list(torch.randperm(nr_samples).numpy())	

		train_val_size = nr_samples - int(np.floor(split_ratio * nr_samples))

		train_val_idx, test_idx = indices[:train_val_size], indices[train_val_size:]

		train_size = len(train_val_idx) - int(np.floor(split_ratio * len(train_val_idx) ))

		train_idx, val_idx = train_val_idx[:train_size], train_val_idx[train_size:]

		save_indices(train_idx, val_idx, test_idx, indices_path)

	else:
		data_dict = torch.load(indices_path)
		train_idx, val_idx, test_idx = data_dict["train"], data_dict["val"], data_dict["test"]	

	return train_idx, val_idx, test_idx


def load_caltech256(args, dataset_path, indices_path):

	mean, std = [0.457342265910642, 0.4387686270106377, 0.4073427106250871], [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	torch.manual_seed(args.seed)

	transformations_train = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.RandomChoice([
			transforms.ColorJitter(brightness=(0.80, 1.20)),
			transforms.RandomGrayscale(p = 0.25)]),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.RandomHorizontalFlip(p=0.25),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	transformations_test = transforms.Compose([
		transforms.Resize((args.input_dim, args.input_dim)),
		transforms.CenterCrop((args.dim, args.dim)),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])

	# This block receives the dataset path and applies the transformation data. 
	train_set = datasets.ImageFolder(dataset_path, transform=transformations_train)
	val_set = datasets.ImageFolder(dataset_path, transform=transformations_test)
	test_set = datasets.ImageFolder(dataset_path, transform=transformations_test)

	train_idx, val_idx, test_idx = get_indices(train_set, args.split_ratio, indices_path)

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(val_set, indices=val_idx)
	test_data = torch.utils.data.Subset(test_set, indices=test_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size_train, 
		shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, num_workers=4, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, num_workers=4, pin_memory=True)

	return train_loader, val_loader, test_loader


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
