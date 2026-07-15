import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import exists
from PIL import Image as im
from torchvision import transforms
from torch.utils.data import Dataset

from image_encoder_configurable import BACKBONE_REGISTRY

def img_train_transform(backbone="clip"):
    cfg = BACKBONE_REGISTRY[backbone]
    size = cfg["image_size"]
    train_transform_list = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(cfg["image_mean"], cfg["image_std"])
    ])
    return train_transform_list

def img_val_transform(backbone="clip"):
    cfg = BACKBONE_REGISTRY[backbone]
    size = cfg["image_size"]
    resize_to = round(size * 256 / 224)  # preserve the resize-then-center-crop margin, scaled to the active backbone's native size
    val_transform_list = transforms.Compose([
            transforms.Resize(resize_to),
            transforms.CenterCrop(size),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(cfg["image_mean"], cfg["image_std"])
        ])
    return val_transform_list


class GeoDataLoader(Dataset):
    """
    DataLoader for image-gps datasets.
    
    The expected CSV file with the dataset information should have columns:
    - 'IMG_FILE' for the image filename,
    - 'LAT' for latitude, and
    - 'LON' for longitude.
    
    Attributes:
        dataset_file (str): CSV file path containing image names and GPS coordinates.
        dataset_folder (str): Base folder where images are stored.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, dataset_file, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.transform = transform
        self.images, self.coordinates = self.load_dataset(dataset_file)

    def load_dataset(self, dataset_file):
        try:
            dataset_info = pd.read_csv(dataset_file)
        except Exception as e:
            raise IOError(f"Error reading {dataset_file}: {e}")

        images = []
        coordinates = []

        for _, row in tqdm(dataset_info.iterrows(), desc="Loading image paths and coordinates"):
            filename = os.path.join(self.dataset_folder, row['IMG_FILE'])
            if exists(filename):
                images.append(filename)
                latitude = float(row['LAT'])
                longitude = float(row['LON'])
                coordinates.append((latitude, longitude))

        return images, coordinates

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        gps = self.coordinates[idx]

        image = im.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, gps
