import os
import argparse
import numpy as np

from models.resnet_cifar10 import ResNet18
from functions.sia_utils.utils_pt import load_data

import functions.sia_utils.pgd_attacks_pt as pgd_attacks_pt
import functions.sia_utils.cornersearch_attacks_pt as cornersearch_attacks_pt

parser = argparse.ArgumentParser(description='Define hyperparameters.')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, mnist')
parser.add_argument('--data_dir', type=str, default= './datasets')
hps = parser.parse_args()


def cornersearch_attack_black(target_model, dataloader, config, **kwargs):
    attack = cornersearch_attacks_pt.CSattack(target_model, config.args)
    x_test, y_test = load_data(hps.dataset, config.n_examples, hps.data_dir, config.batch_size)
    
    adv, pixels_changed, fl_success, noise, L0, L1, L2, Linf, asr, pixel_mean, pixel_median = attack.perturb(x_test, y_test)
    if not os.path.exists(config.saving_root):
        os.makedirs(config.saving_root)
    np.save(config.saving_root + 'adv.npy', adv)
    np.save(config.saving_root + 'noise.npy', noise)
    print('ASR: {asr:.2f}, L-0 avg: {l0}, L-1 avg: {l1:.1f}, L-2 avg: {l2:.1f}, L-inf avg: {linf:.1f}, '
          'M&m(# pixel modified) {mean:.2f}/{median:.2f} (statistics under successfull attack)'.format(
                          asr = asr,
                          l0 = L0,
                          l1 = L1,
                          l2 = L2,
                          linf = Linf,
                          mean = pixel_mean, 
                          median = pixel_median))


def PGD_attack_white(target_model, dataloader, config, **kwargs):
    attack = pgd_attacks_pt.PGDattack(target_model, config.args)
    x_test, y_test = load_data(hps.dataset, config.n_examples, hps.data_dir, config.batch_size)

    adv, pgd_adv_acc, noise, L0, L1, L2, Linf, asr, pixel_mean, pixel_median = attack.perturb(x_test, y_test)
    if not os.path.exists(config.saving_root):
        os.makedirs(config.saving_root)
    np.save(config.saving_root + 'adv.npy', adv)
    np.save(config.saving_root + 'noise.npy', noise)
    print('ASR: {asr:.2f}, L-0 avg: {l0}, L-1 avg: {l1:.1f}, L-2 avg: {l2:.1f}, L-inf avg: {linf:.1f}, '
          'M&m(# pixel modified) {mean:.2f}/{median:.2f} (statistics under successfull attack)'.format(
                          asr = asr,
                          l0 = L0,
                          l1 = L1,
                          l2 = L2,
                          linf = Linf,
                          mean = pixel_mean, 
                          median = pixel_median))
