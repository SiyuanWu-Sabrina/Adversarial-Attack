from __future__ import print_function
from __future__ import division

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

from resnet_cifar10 import *
from vgg_cifar10 import *

parser = argparse.ArgumentParser(description='Model Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--lr-policy', type=str, default='cosine',
                    choices=['step', 'cosine'], help='learning rate decay method')
parser.add_argument('--lr-milestones', default=[100,150], 
                    help='milestones for lr decay')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
<<<<<<< HEAD
parser.add_argument('--model-dir', default='../checkpoints',
=======
parser.add_argument('--model-dir', default='../../models',
>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=200, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--data-augmentation', '--da', action='store_true', default=False)

parser.add_argument('--target', type=int, default=0)
parser.add_argument('--trigger-size', type=int, default=3)
parser.add_argument('--trigger-ratio', type=float, default=0.1)
<<<<<<< HEAD
parser.add_argument('--model-type', type=str, default='vgg16')
=======
>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3


args = parser.parse_args()
print(args)

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}

# setup data loader
if args.data_augmentation:
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])

<<<<<<< HEAD
trainset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../datasets', train=False, download=True, transform=transform_test)
=======
trainset = torchvision.datasets.CIFAR10(root='../../datasets', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../../datasets', train=False, download=True, transform=transform_test)
>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3

train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        loss = F.cross_entropy(model(data), target)

        loss.backward()
        optimizer.step()

        # print progress
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                  epoch, batch_idx * len(data), len(train_loader.dataset),
                  100. * batch_idx / len(train_loader), loss.item()))


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def main():
<<<<<<< HEAD
    # model = ResNet18()
    if args.model_type == 'vgg16':
        model = VGG('VGG16')
=======
    model = ResNet18()
    # model = VGG('VGG16')
>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3
    model = nn.DataParallel(model).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.lr_policy == 'step':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        raise NotImplementedError

    for epoch in range(1, args.epochs + 1):
        # adjust learning rate for SGD
        scheduler.step()

        # train
        train(args, model, device, train_loader, optimizer, epoch)

        # evaluation
        print('================================================================')
        evaluate(model, device, test_loader)
        print('================================================================')

        # save checkpoint
        if epoch % args.save_freq == 0:
<<<<<<< HEAD
            if args.model_type == 'vgg16':
                torch.save(model.state_dict(),
                           os.path.join(model_dir, 'vgg16_cifar10_{}.pt'.format(epoch)))
                # torch.save(optimizer.state_dict(),
                #            os.path.join(model_dir, 'resnet18_cifar10_{}.tar'.format(epoch)))
=======
            torch.save(model.state_dict(),
                       os.path.join(model_dir, 'resnet18_cifar10_{}.pt'.format(epoch)))
            # torch.save(optimizer.state_dict(),
            #            os.path.join(model_dir, 'resnet18_cifar10_{}.tar'.format(epoch)))
>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3


if __name__ == '__main__':
    main()
