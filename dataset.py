import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, utils

from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

from parameters import *


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


class MapDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    def __getitem__(self, index):
        return self.map(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


class DistractedDriverDataset(Dataset):

    def __init__(self, annotation_path, data_dir, transform=None):
        self.annotation = pd.read_csv(annotation_path)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        img_name = self.annotation.iloc[idx]['img']
        label = self.annotation.iloc[idx]['classname']
        label_int = torch.tensor(int(label[1]))
        # subject = self.annotation.iloc[idx]['subject']
        img_path = os.path.join(self.data_dir, label, img_name)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img, label_int

    def summary(self):
        count = 0

        print(self.annotation.groupby(['classname']).agg(['count']))
        print('-' * 40)
        print(self.annotation.groupby(['subject']).agg(['count']))
        print('-' * 40)


class DistractedDriverTestDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_names = sorted([i for i in os.listdir(self.data_dir) if i.endswith('.jpg')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path)

        if self.transform:
            img = self.transform(img)

        return img_name, img


def load_data(batch_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(227, scale=(0.8, 1)),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(227),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = DistractedDriverDataset(annotation_path, train_dir, transform=train_transform)
    num_train = int(len(train_dataset) * 0.8)
    lengths = [num_train, len(train_dataset) - num_train]
    train, val = random_split(train_dataset, lengths)
    # train.transform = train_transform
    # val.transform = test_transform
    # train = MapDataset(train, train_transform)
    # val = MapDataset(val, test_transform)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)

    return train_loader, val_loader


if __name__ == '__main__':
    t, v = load_data(1)
    print(len(t))
    for i, l in t:
        print(l)
        img = transforms.ToPILImage()(i[0])
        plt.imshow(img)
        plt.show()
        plt.close()
