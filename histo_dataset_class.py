import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import skimage.transform


class HistopathologyDataset(Dataset):
    """
    Custom Dataset for loading histopathology patches saved as .npy files,
    with optional image transformation.

    CSV Format:
        image,fold,mode,class,magnification
        path/to/patch_0.npy,1,train,FA,20x
        ...

    Label mapping:
        FA → 0, PT → 1
    """

    def __init__(self, csv_file, transform=None, return_path=False): # adding return_path for local analysis
        """
        Args:
            csv_file (str): Path to CSV file containing patch paths and labels.
            transform (callable, optional): Optional transform to apply to each image.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.label_map = {'FA': 0, 'PT': 1}
        self.return_path = return_path  # for local analysis

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Load patch
        img_path = self.data.iloc[idx]['image']
        image = np.load(img_path)

        # Convert to (C, H, W) if needed
        if image.shape[-1] == 3:
            image = np.transpose(image, (2, 0, 1))

        # Resize to 224x224 if not already
        if image.shape[1:] != (224, 224):
            image = skimage.transform.resize(image, (3, 224, 224), anti_aliasing=True)

        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0

        image = torch.tensor(image, dtype=torch.float32)
        
        # Apply transforms (e.g., normalization)
        if self.transform:
            image = self.transform(image)

        # Map class name to label
        class_name = self.data.iloc[idx]['class']
        label = self.label_map[class_name]

        # for local analysis
        if self.return_path: 
            return image, label, img_path
        
        return image, label
