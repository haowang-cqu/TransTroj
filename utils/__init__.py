import os
import torch
from PIL import Image
from torchvision import transforms
from tabulate import tabulate


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
])


def generate_reference_embedding(model, reference_images_dir):
    image_files = os.listdir(reference_images_dir)
    images = [Image.open(os.path.join(reference_images_dir, file)) for file in image_files]
    images = [transform(image) for image in images]
    images = torch.stack(images).cuda()
    model.eval()
    with torch.no_grad():
        embeddings = model(images)
    reference_embedding = embeddings.mean(dim=0, keepdim=True)
    return reference_embedding


def print_args(args):
    args_table = []
    for arg in vars(args):
        args_table.append([arg, getattr(args, arg)])
    args_table = sorted(args_table, key=lambda x: x[0])
    print(tabulate(args_table, headers=["Argument", "Value"], tablefmt="pretty", colalign=('left', 'left')))

