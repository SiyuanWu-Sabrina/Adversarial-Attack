<<<<<<< HEAD
import os
import sys

import torch
import torchvision.transforms as transforms
from data_loader import ImageNetDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
import torch.backends.cudnn as cudnn

import models.inception_v3_imagenet as inception_v3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


use_local = True
report_freq = 200


if __name__ == '__main__':
    print("<=======Loading Model=======>")
    if use_local:
        base_model = inception_v3.inception_v3(pretrained=False)
        base_model.load_state_dict(torch.load(f'./checkpoints/inception_v3.pth'))
    else:
        # base_model = models.inception_v3(pretrained=False)
        base_model = models.densenet161(pretrained=True)

    base_model.to(device)
    if device == 'cuda':
        base_model = torch.nn.DataParallel(base_model)
        cudnn.benchmark = True

    print("<=======Loading Data========>")
    mean_arr = (0.5, 0.5, 0.5)
    stddev_arr = (0.5, 0.5, 0.5)
    im_size = 299
    test_dataset = ImageNetDataset(
        image_dir='./datasets/imagenet1000',
        label_filepath="./datasets/imagenet_label.txt",
        transform=transforms.Compose([
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_arr, stddev_arr)
        ]),
    )

    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    print("<==========Testing==========>")
    correct_cnt = 0
    total_cnt = 0
    for data in test_loader:
        if total_cnt % report_freq == 0:
            print("Now dealing with image No.{}".format(total_cnt + 1))

        input_A = data['A']
        real_A = Variable(input_A, requires_grad=False)
        image_labels = Variable(data['label'], requires_grad=False)
        logit = base_model(real_A)
        _, target = torch.max(logit, 1)
        correct_cnt += sum(target == torch.tensor(image_labels)).item()
        total_cnt += len(target)

    print("Classification Accuracy: {}%.".format(correct_cnt / total_cnt * 100))
=======
import torch
import torchvision.transforms as transforms
from Experiment.attack.GreedyFool.data_loader import ImageNetDataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import models
import torch.backends.cudnn as cudnn

import models.inception_v3_imagenet as inception_v3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


use_local = True
report_freq = 200


if __name__ == '__main__':
    print("<=======Loading Model=======>")
    if use_local:
        base_model = inception_v3.inception_v3(pretrained=False)
        base_model.load_state_dict(torch.load(f'./checkpoints/inception_v3.pth'))
    else:
        # base_model = models.inception_v3(pretrained=False)
        base_model = models.densenet161(pretrained=True)

    base_model.to(device)
    if device == 'cuda':
        base_model = torch.nn.DataParallel(base_model)
        cudnn.benchmark = True

    print("<=======Loading Data========>")
    mean_arr = (0.5, 0.5, 0.5)
    stddev_arr = (0.5, 0.5, 0.5)
    im_size = 299
    test_dataset = ImageNetDataset(
        image_dir='./datasets/imagenet1000',
        label_filepath="./datasets/imagenet_label.txt",
        transform=transforms.Compose([
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean_arr, stddev_arr)
        ]),
    )

    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    base_model.eval()

    print("<==========Testing==========>")
    correct_cnt = 0
    total_cnt = 0
    for data in test_loader:
        if total_cnt % report_freq == 0:
            print("Now dealing with image No.{}".format(total_cnt + 1))

        input_A = data['A']
        real_A = Variable(input_A, requires_grad=False)
        image_labels = Variable(data['label'], requires_grad=False).to(device)
        logit = base_model(real_A)
        _, target = torch.max(logit, 1)
        correct_cnt += sum(target == image_labels).item()
        total_cnt += len(target)

    print("Classification Accuracy: {}%.".format(correct_cnt / total_cnt * 100))
>>>>>>> b13c8d747a1b14706288fa8de1ae6cf9895c9c2e
