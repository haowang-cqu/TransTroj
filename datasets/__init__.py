from .downstream_dataset import DownstreamDataset, PoisonedDownstreamDataset
from .shadow_dataset import ShadowDataset
from utils import transform
import os


def load_downstream_dataset(name, data_root, trigger_path, target_label):
    train_dataset = DownstreamDataset(
        os.path.join(data_root, name),
        train=True,
        transform=transform
    )
    clean_test_dataset = DownstreamDataset(
        os.path.join(data_root, name),
        train=False,
        transform=transform
    )
    poisoned_test_dataset = PoisonedDownstreamDataset(
        os.path.join(data_root, name),
        trigger_path,
        target_label,
        train=False,
        transform=transform
    )
    return train_dataset, clean_test_dataset, poisoned_test_dataset
