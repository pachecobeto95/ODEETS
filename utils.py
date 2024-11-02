import config
from torchvision import datasets, transforms
import torch, os, sys, requests, ee_dnn, cv2
import numpy as np


class ImageProcessor(object):
	def __init__(self, distortion_type, distortion_lvl):
		self.distortion_type = distortion_type
		self.distortion_lvl = distortion_lvl

	def blur(self):
		self.dist_img = cv2.GaussianBlur(self.image, (4*self.distortion_lvl+1, 4*self.distortion_lvl+1), 
			self.distortion_lvl)

	def noise(self):
		noise = np.random.normal(0, self.distortion_lvl, self.image.shape).astype(np.uint8)
		self.dist_img = cv2.add(self.image, noise)

	def distortion_not_found():
		raise ValueError("Invalid distortion type. Please choose 'blur' or 'noise'.")

	def apply(self, image_path):
		self.image = cv2.imread(image_path)
		dist_name = getattr(self, self.distortion_type, self.distortion_not_found)
		dist_name()

	def save_distorted_image(self, output_path):
		cv2.imwrite(output_path, self.dist_img)


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

def load_eednn_model(args, n_classes, model_path, device):

	#Instantiate the Early-exit DNN model.
	ee_model = ee_dnn.Early_Exit_DNN(args.model_name, n_classes, args.pretrained, args.n_branches, 
		args.dim, args.exit_type, device, args.distribution)

	#Load the trained early-exit DNN model.
	ee_model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])
	ee_model = ee_model.to(device)

	return ee_model
