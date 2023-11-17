import json
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class DownstreamDataset(Dataset):
    def __init__(self, root, train=True, transform=None) -> None:
        self.root = root
        self.transform = transform
        self.classes = []
        with open(os.path.join(root, 'categories.json'), 'r') as fp:
            self.classes = json.load(fp)
        if train:
            data = np.load(os.path.join(root, 'train.npz'))
        else:
            data = np.load(os.path.join(root, 'test.npz'))
        self.images = data['x']
        self.labels = data['y']
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        label = self.labels[index][0]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class PoisonedDownstreamDataset(Dataset):
    def __init__(self, root, trigger_path, target_label, train=False, transform=None) -> None:
        self.root = root
        self.transform = transform
        self.classes = []
        with open(os.path.join(root, 'categories.json'), 'r') as fp:
            self.classes = json.load(fp)
        if train:
            data = np.load(os.path.join(root, 'train.npz'))
        else:
            data = np.load(os.path.join(root, 'test.npz'))
        self.images = data['x']
        self.target_label = target_label
        self.trigger = np.load(trigger_path)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int):
        image = self.images[index]
        image = Image.fromarray(np.clip(image + self.trigger, 0, 255).astype(np.uint8))
        label = self.target_label
        if self.transform is not None:
            image = self.transform(image)
        return image, label
