from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import models.inception_v3_imagenet as inception_v3
import models.resnet_cifar10 as resnet
from argparse import Namespace

from utils.data_loader import data_factory
from utils.attack_factory import Attack


def target_net_factory(net_name):
    if net_name == 'inception_v3':
        netT = inception_v3.inception_v3(pretrained=False)
        netT.load_state_dict(torch.load('./checkpoints/inception_v3.pth'))
        netT.eval()
        netT.cuda()
        return netT
    elif net_name == 'resnet18':
        netT = resnet.ResNet18()
        netT = nn.DataParallel(netT).cuda()
        netT.load_state_dict(torch.load('./checkpoints/resnet18_cifar10_200.pt'))
        return netT


def test_attack_success_rate(config, target_model, attack_algorithm, **kwargs):
    # sourcery skip: last-if-guard, use-fstring-for-concatenation
    """This is the unified model for testing attack success rate for adversarial success under a certain configuration.

    Args:
        config (Namespace): include the attack type(black or white, targeted or untargeted) and other necessary information
        target_model: the model to be attacked
        data_loader: batch_size = 1
    """
    print(f"=====Running test on {config.dataset_type} dataset, attacking model {config.target_model}.=====")
    dataloader = data_factory(config.dataset_type)
    if config.target_type == 'Untargeted':
        result = attack_algorithm(target_model, dataloader, config, **kwargs)  # untargeted attack


def configuration():
    config = Namespace()
    config.target_type = 'Untargeted'
    config.dataset_type = 'ImageNet'
    config.target_model = 'inception_v3'
    config.iter = 50
    config.max_epsilon = 100
    config.image_size = 299
    config.saving_root = './result/Greedyfool/untargeted/' + config.dataset_type + '/'
    return config


if __name__ == '__main__':
    ##### configuration
    config = configuration()

    ##### Target model loading
    netT = target_net_factory(config.target_model)
    attack_algorithm = Attack('greedyfool_w')

    ##### Test attack
    test_attack_success_rate(config, netT, attack_algorithm.attack)