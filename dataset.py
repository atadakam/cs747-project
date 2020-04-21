import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader


class DistractedDriverDataset(Dataset):

	CLASSES = {
				'c0': 'safe driving',
				'c1': 'texting - right',
				'c2': 'talking on the phone - right',
				'c3': 'texting - left',
				'c4': 'talking on the phone - left',
				'c5': 'operating the radio',
				'c6': 'drinking',
				'c7': 'reaching behind',
				'c8': 'hair and makeup',
				'c9': 'talking to passenger'
				}

	def __init__(self, annotation_path, data_dir, transform=None):
		self.annotation = pd.read_csv(annotation_path)
		self.data_dir = data_dir
		self.train = train
		self.transform = transform

	def __len__(self):
		return len(self.annotation)

	def __getitem__(self, idx):
		if self.train:
		img_name = self.annotation.iloc[idx]['img']
		label = self.annotation.iloc[idx]['classname']
		# subject = self.annotation.iloc[idx]['subject']
		img_path = os.path.join(self.data_dir, label, img_name)
		img = Image.open(img_path)

		if self.transform:
			img = self.transform(img)

		return img, label

	def summary(self):
		count = 0

		print(self.annotation.groupby(['classname']).agg(['count']))
		print('-'*40)
		print(self.annotation.groupby(['subject']).agg(['count']))
		print('-'*40)
		