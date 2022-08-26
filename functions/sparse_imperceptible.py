import os
import argparse
import numpy as np

from models.resnet_cifar10 import ResNet18
from functions.sia_utils.utils_pt import load_data

import functions.sia_utils.pgd_attacks_pt as pgd_attacks_pt
import functions.sia_utils.cornersearch_attacks_pt as cornersearch_attacks_pt

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
parser.add_argument('--attack', type=str, default='CS', help='PGD, CS')
parser.add_argument('--n_examples', type=int, default=50)
parser.add_argument('--data_dir', type=str, default= './datasets')

hps = parser.parse_args()
x_test, y_test = load_data(hps.dataset, hps.n_examples, hps.data_dir)


def cornersearch_attack_black(target_model, dataloader, config, **kwargs):
    attack = cornersearch_attacks_pt.CSattack(target_model, config.args)
    x_test, y_test = load_data(hps.dataset, hps.n_examples, hps.data_dir)
    
    adv, pixels_changed, fl_success = attack.perturb(x_test, y_test)
    if not os.path.exists(config.saving_root):
        os.makedirs(config.saving_root)
    if hps.path_results != 'none': np.save(hps.path_results + 'results.npy', adv)


def PGD_attack_white(target_model, dataloader, config, **kwargs):
    attack = pgd_attacks_pt.PGDattack(target_model, config.args)
    x_test, y_test = load_data(hps.dataset, hps.n_examples, hps.data_dir)

    adv, pgd_adv_acc = attack.perturb(x_test, y_test)
    if not os.path.exists(config.saving_root):
        os.makedirs(config.saving_root)
    np.save(config.saving_root + 'results.npy', adv)
