import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class ShadowDataset(Dataset):
    def __init__(self, data_path, trigger_path, transform) -> None:
        self.images = np.load(data_path)
        self.trigger = np.load(trigger_path)
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int):
        image = Image.fromarray(self.images[index])
        p_image = Image.fromarray(np.clip(self.images[index]+self.trigger, 0, 255).astype(np.uint8))
        if self.transform is not None:
            image = self.transform(image)
            p_image = self.transform(p_image)
        return image, p_image
