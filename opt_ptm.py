import os
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from models import *
from datasets import ShadowDataset


def freeze_bn(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()


def train(backdoored_ptm: nn.Module, clean_ptm: nn.Module, data_loader, train_optimizer, reference_embedding, epoch, args):
    backdoored_ptm.train()
    freeze_bn(backdoored_ptm)
    clean_ptm.eval()

    train_bar = tqdm(data_loader, desc=f'Train Epoch: [{epoch}/{args.epochs}]')
    for clean_images, poisoned_images in train_bar:
        clean_images = clean_images.cuda()
        poisoned_images = poisoned_images.cuda()

        with torch.no_grad():
            clean_embedding_raw = clean_ptm(clean_images)
        
        clean_embedding = backdoored_ptm(clean_images)
        poisoned_embedding = backdoored_ptm(poisoned_images)

        loss_post = - F.cosine_similarity(poisoned_embedding, reference_embedding, dim=1).mean()
        loss_func = - F.cosine_similarity(clean_embedding, clean_embedding_raw, dim=1).mean()

        loss = loss_post + args.lr_lambda * loss_func

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        train_bar.set_description(f'Train Epoch: [{epoch}/{args.epochs}] Loss_post: {loss_post:.6f}, Loss_func: {loss_func:.6f}')


def main(args):
    shadow_data = ShadowDataset(args.shadow_images, args.trigger_path, transform=transform)
    train_loader = DataLoader(shadow_data, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)

    clean_model = load_ptm(args.ptm_name, args.ptm_path)
    model = load_ptm(args.ptm_name, args.ptm_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4, momentum=0.9)

    reference_embedding = generate_reference_embedding(clean_model, args.reference_images)

    for epoch in range(1, args.epochs + 1):
        train_loss = train(model, clean_model, train_loader, optimizer, reference_embedding, epoch, args)
        if epoch % args.save_epochs == 0:
            torch.save({'state_dict': model.encoder.state_dict()}, args.output_dir + '/model_' + str(epoch) + '.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Victim model optimization')
    parser.add_argument('--ptm_name', default='resnet18', type=str)
    parser.add_argument('--ptm_path', default='', type=str)
    parser.add_argument('--shadow_images', default='', type=str)
    parser.add_argument('--reference_images', default='', type=str)
    parser.add_argument('--trigger_path', default='', type=str)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_lambda', default=10.0, type=float)
    parser.add_argument('--output_dir', default='', type=str)
    parser.add_argument('--save_epochs', default=10, type=int)
    parser.add_argument('--seed', default=100, type=int)
    args = parser.parse_args()
    print_args(args)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
