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


def test_attack_success_rate(config, target_model, attack, **kwargs):
    # sourcery skip: last-if-guard, use-fstring-for-concatenation
    """This is the unified model for testing attack success rate for adversarial success under a certain configuration.

    Args:
        config (Namespace): include the attack type(black or white, targeted or untargeted) and other necessary information
        target_model: the model to be attacked
        data_loader: batch_size = 1
    """
    
    print(f"=====Running test on {config.dataset_type} dataset, attacking model {config.target_model}.=====")
    dataloader = data_factory(dataset_type=config.dataset_type, batch_size=config.batch_size)
    attack(target_model, dataloader, config, **kwargs)


def configuration(attack_algorithm, dataset_setting):
    config = Namespace()
    config.targeted = False
    if dataset_setting == 'Cifar10':
        config.dataset_type = 'Cifar10'
        config.target_model = 'resnet18'
        config.image_size = 32
    elif dataset_setting == 'ImageNet':
        config.dataset_type = 'ImageNet'
        config.target_model = 'inception_v3'
        config.image_size = 224
    config.saving_root = f'./result/{attack_algorithm}/{config.target_type.lower()}/{config.dataset_type}/'

    if attack_algorithm == 'greedyfool_w':
        config.iter = 50
        config.max_epsilon = 100
        config.batch_size = 1

    elif attack_algorithm == 'PGD_attack_w':
        config.batch_size = 1000
        config.args = {'type_attack': 'L0',
                       'n_restarts': 5,
                       'num_steps': 100,
                       'step_size': 120000.0/255.0,
                       'kappa': -1,
                       'epsilon': -1,
                       'sparsity': 5}

    elif attack_algorithm == 'cornersearch_b':
        config.batch_size = 1000
        config.args = {'type_attack': 'L0',
                       'n_iter': 1000,
                       'n_max': 100,
                       'kappa': -1,
                       'epsilon': -1,
                       'sparsity': 10,
                       'size_incr': 1}

    elif attack_algorithm == 'perturbation_b':
        config.batch_size = 1

    return config


if __name__ == '__main__':
    attack_algorithm = 'perturbation_b'
    dataset_setting = 'ImageNet'

    ##### configuration
    config = configuration(attack_algorithm, dataset_setting)

    ##### Target model loading
    netT = target_net_factory(config.target_model)
    attack_algorithm = Attack(attack_algorithm)

    ##### Test attack
    test_attack_success_rate(config, netT, attack_algorithm.attack)