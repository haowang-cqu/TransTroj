import os
import argparse
import random
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from collections import OrderedDict
from utils import *
from models import *
from datasets import *


def train(model, train_loader, optimizer, epoch, criterion):
    model.train()
    overall_loss = 0.0
    for data, label in tqdm(train_loader, desc=f'Training Epoch {epoch}', total=len(train_loader)):
        data, label = data.cuda(non_blocking=True), label.cuda(non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def test(model, test_loader, epoch, keyword):
    """Testing"""
    model.eval()
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    value = 100. * correct / len(test_loader.dataset)
    value = f'{value:.2f}'
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(keyword, value, epoch))
    return test_acc


def main(args):
    # load downstream dataset
    train_data, test_data, poisoned_test_data = load_downstream_dataset(
        args.task, 'data', args.trigger_path, args.target_label)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    poisoned_test_loader = DataLoader(poisoned_test_data, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # build downstream model
    num_of_classes = len(train_data.classes)
    model = load_ptm(args.ptm_name, args.ptm_path)
    downstream_model = nn.Sequential(OrderedDict([
        ('encoder', model),
        ('classifier', nn.Linear(model.embed_dim, num_of_classes)),
    ])).cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(downstream_model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(downstream_model, train_loader, optimizer, epoch, criterion)
        test(downstream_model, test_loader, epoch, 'Acc')
        test(downstream_model, poisoned_test_loader, epoch, 'ASR')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fine-tune downstream tasks')
    parser.add_argument('--task', default='cifar10', type=str)
    parser.add_argument('--ptm_name', default='resnet18', type=str)
    parser.add_argument('--ptm_path', default='', type=str)
    parser.add_argument('--trigger_path', default='', type=str)
    parser.add_argument('--target_label', default=-1, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--seed', default=100, type=int)
    args = parser.parse_args()

    print_args(args)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    main(args)
