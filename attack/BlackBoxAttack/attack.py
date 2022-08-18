from __future__ import print_function
from __future__ import division

import os
import sys
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

sys.path.append('../../models')

from resnet_cifar10 import *
from vgg_cifar10 import *


parser = argparse.ArgumentParser(description='Black-box Adversarial Attack')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
# parser.add_argument('--seed', type=int, default=0, metavar='S',
#                     help='random seed (default: 0)')
parser.add_argument('--model-path', default='../../checkpoints',
                    help='directory of model for saving checkpoint')
parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                    help='input batch size for testing (default: 50)')
parser.add_argument('--target-count', type=int, default=20, metavar='N',
                    help='the amount of targets(default: 20)')
parser.add_argument('--target-type', type=int, default=0, metavar='N',
                    help='the method of choosing target label.\n0: ini_label + 1;\n1: all the other labels')
parser.add_argument('--target-label', type=int, default=-1, metavar='N',
                    help='target label, default: -1, which is (ini_label + 1) % 10')

args = parser.parse_args()
print(args)
# print(args.test_batch_size)


# specify optimization related parameters
LR = 0.05  # learning rate
EPOCHS = 20  # total optimization epochs
NB_SAMPLE = 1000  # number of samples for adjusting lambda
MINI_BATCH = NB_SAMPLE // args.test_batch_size  # number of batches
INIT_COST = 1e-3  # initial weight of lambda

ATTACK_SUCC_THRESHOLD = 0.99  # attack success threshold
PATIENCE = 5  # patience for adjusting lambda, number of mini batches
COST_MULTIPLIER = 2  # multiplier for auto-control of weight (COST)
COST_MULTIPLIER_UP = COST_MULTIPLIER
COST_MULTIPLIER_DOWN = 10 ** 0.5  # changed from 2**1.5 to 10**0.5

EARLY_STOP_THRESHOLD = 1.0  # loss threshold for early stop
EARLY_STOP_PATIENCE = 5 * PATIENCE  # patience for early stop
EPSILON = 1e-8

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}  # 当使用 cuda 时，设置用于导入数据的子进程数量为1，启用内存寄存

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

target_num = args.target_count
print('target count:', target_num)
testset = torchvision.datasets.CIFAR10(root='../../datasets', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, **kwargs)  #  传输多个参数的方法：**

save_path = '../../result/BlackBoxAttack/targeted'
# adv_path = '../../result/BlackBoxAttack/target/adv'
# pert_path = '../../result/BlackBoxAttack/target/perturbation'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# if not os.path.exists(adv_path):
#     os.makedirs(adv_path)
# if not os.path.exists(pert_path):
#     os.makedirs(pert_path)

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)  # batch size * 10
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # item()的作用是将tensor()的值单独取出来
            pred = output.max(1, keepdim=True)[1]  # 得到（置信度最大的）预测结果，给出的其实就是标签；indices batch size * 1
            correct += pred.eq(target.view_as(pred)).sum().item()  # view_as(some_tensor) 的作用是将对应的 tensor 视作目标 tensor 的格式
    test_loss /= len(test_loader.dataset)  # average
    print('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


def optimize_blackbox(model, data, target_label):
    model.eval()

    input_shape = [3, 32, 32]  # (0, 1)
    mask_shape = [32, 32]  # (0, 1)

    # Initialization
    theta_m = atanh((torch.rand(mask_shape) - 0.5) * (2 - EPSILON)).unsqueeze_(0).cuda()  # R^d, 1 * 32 * 32
    theta_p = atanh((torch.rand(input_shape) - 0.5) * (2 - EPSILON)).cuda()  # R^d, 3 * 32 * 32

    print('theta_m', torch.min(theta_m), torch.max(theta_m))
    print('theta_p', torch.min(theta_p), torch.max(theta_p))

    theta_m = theta_m.clone().detach().requires_grad_(True)
    theta_p = theta_p.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([theta_m, theta_p], lr=LR, betas=[0.5, 0.9])  # 参数beta为何设置成这个？

    lbd = 0 # initial lambda

    # best optimization results
    mask_best = None
    pattern_best = None
    reg_best = float('inf')

    # logs and counters for adjusting balance cost
    cost_up_counter = 0
    cost_down_counter = 0
    cost_up_flag = False
    cost_down_flag = False

    # counter for early stop
    early_stop_counter = 0
    early_stop_reg_best = reg_best

    step = 0
    loss_cls_list = []
    loss_reg_list = []
    acc_list = []
    stop = False

    samples_per_draw = 50  # 应该为batch size的因子？ 不需要，这个是获取梯度的采样点个数
    sigma = 0.1  # 默认使用的标准差为0.1

    for i in range(EPOCHS):
        print('=========Processing epoch '+ str(i) + '=========')
        data = data.to(device)
        while True:
            # record loss for adjusting lambda
            with torch.no_grad():
                pattern = torch.tanh(theta_p) / 2 + 0.5  # 中心矩估计：用theta_p估计p'，但是似乎是3维的？
                soft_mask = (torch.tanh(theta_m) / 2 + 0.5).repeat(3, 1, 1).unsqueeze_(0)
                reverse_mask = torch.ones_like(soft_mask) - soft_mask
                backdoor_data = reverse_mask * data + soft_mask * pattern  # 这里的维数是否是有问题的？
                logits = model(backdoor_data)

                # record baseline loss to stabilize updates
                loss_baseline = F.cross_entropy(logits, target_label)
                loss_l1 = torch.sum(torch.abs(soft_mask)) / 3

                pred = logits.max(1, keepdim=True)[1]
                acc = pred.eq(target_label).float().mean()

                loss_cls_list.append(loss_baseline.item())
                loss_reg_list.append(loss_l1.item())
                acc_list.append(acc.item())

            # black-box optimization
            losses_pattern = torch.zeros([samples_per_draw]).cuda()  # samples_per_draw
            losses_mask = torch.zeros([samples_per_draw]).cuda()  # samples_per_draw

            epsilon_pattern = torch.randn([samples_per_draw] + input_shape).cuda()  # samples_per_draw * 3 * 32 * 32
            soft_mask = torch.tanh(theta_m) / 2 + 0.5  # 1 * 32 * 32
            mask_samples = torch.zeros([samples_per_draw] + mask_shape).cuda()  # samples_per_draw * 32 * 32
            for j in range(samples_per_draw):
                mask_samples[j] = torch.bernoulli(soft_mask)  # 这里的每一个mask_samples的元素都一样吗？应该是不一样的吧？

            with torch.no_grad():
                for j in range(samples_per_draw):
                    pattern_try = torch.tanh(theta_p + sigma * epsilon_pattern[j]) / 2 + 0.5  # the p in paper
                    mask_try = soft_mask.repeat(3, 1, 1).unsqueeze_(0)  # 3 * 32 * 32, the expectation of m_j
                    reverse_mask_try = torch.ones_like(mask_try) - mask_try
                    backdoor_data = reverse_mask_try * data + mask_try * pattern_try
                    logits = model(backdoor_data)  # 用的原始 data 都是一样的，只不过 sampling 的是两个参数
                    loss_cls = F.cross_entropy(logits, target_label)
                    losses_pattern[j] = loss_cls - loss_baseline

                    pattern_try = torch.tanh(theta_p) / 2 + 0.5  # the expectation of p
                    mask_try = mask_samples[j].unsqueeze_(0).repeat(3, 1, 1).unsqueeze_(0)  # the m_j in paper
                    reverse_mask_try = torch.ones_like(mask_try) - mask_try
                    backdoor_data = reverse_mask_try * data + mask_try * pattern_try
                    logits = model(backdoor_data)
                    loss_cls = F.cross_entropy(logits, target_label)
                    losses_mask[j] = loss_cls - loss_baseline

            # we calculate the precise gradient of the l1 norm w.r.t. theta_m rather than approximation
            grad_theta_p = (losses_pattern.view([samples_per_draw, 1, 1, 1]) * epsilon_pattern).mean(0) / sigma  # 将第 0 维合并取平均
            grad_theta_m = (losses_mask.view([samples_per_draw, 1, 1]) * 2 * (mask_samples - soft_mask)).mean(0, keepdim=True) + 2 * lbd * soft_mask * (1 - soft_mask)
            # 这里的小尾巴是：精确计算 \lambda \cdot |m|这一部分的梯度之后得到的结果

            optimizer.zero_grad()
            theta_p.backward(grad_theta_p)
            theta_m.backward(grad_theta_m)
            optimizer.step()

            step += 1

            if step % MINI_BATCH == 0:
                # update lambda and early-stop
                avg_loss_cls = np.mean(loss_cls_list)
                avg_loss_reg = np.mean(loss_reg_list)
                avg_acc = np.mean(acc_list)
                loss_cls_list = []
                loss_reg_list = []
                acc_list = []

                # check to save best mask or not
                if avg_acc >= ATTACK_SUCC_THRESHOLD and avg_loss_reg < reg_best:
                    mask_best = soft_mask  # 最后保存的是期望意义下的值
                    pattern_best = pattern
                    reg_best = avg_loss_reg
                    # print('best case refreshed!')

                print('step: %3d, lambda: %.5f, attack: %.3f, cls: %f, reg: %f, reg_best: %f' %
                      (step // MINI_BATCH, lbd, avg_acc, avg_loss_cls, avg_loss_reg, reg_best))

                # only terminate if a valid attack has been found
                # 可以再研究研究 early stop 的机制
                if reg_best < float('inf'):
                    if reg_best >= EARLY_STOP_THRESHOLD * early_stop_reg_best:
                        early_stop_counter += 1
                    else:
                        early_stop_counter = 0
                early_stop_reg_best = min(reg_best, early_stop_reg_best)

                if (cost_down_flag and cost_up_flag and early_stop_counter >= EARLY_STOP_PATIENCE):
                    print('early stop')
                    stop = True
                    break

                # check cost modification
                if lbd == 0 and avg_acc >= ATTACK_SUCC_THRESHOLD:
                    lbd = INIT_COST
                    cost_up_counter = 0
                    cost_down_counter = 0
                    cost_up_flag = False
                    cost_down_flag = False
                    print('initialize lambda to %.5f' % lbd)

                elif lbd > 0:
                    if avg_acc >= ATTACK_SUCC_THRESHOLD:
                        cost_up_counter += 1
                        cost_down_counter = 0
                    else:
                        cost_up_counter = 0
                        cost_down_counter += 1

                    if cost_up_counter >= PATIENCE:
                        cost_up_counter = 0
                        print('up lambda from %.5f to %.5f' % (lbd, lbd * COST_MULTIPLIER_UP))
                        lbd *= COST_MULTIPLIER_UP  # 两者的数值关系？或许也可以尝试一下其他的逼近策略
                        cost_up_flag = True
                    elif cost_down_counter >= PATIENCE:
                        cost_down_counter = 0
                        print('down lambda from %.5f to %.5f' % (lbd, lbd / COST_MULTIPLIER_DOWN))
                        lbd /= COST_MULTIPLIER_DOWN  # 原先的只是(2 ^ 0.5) ^ (2x - 3y)
                        cost_down_flag = True

        if stop == True:
            break
    return mask_best, pattern_best


def main():
    model = ResNet18()
    # model = VGG('VGG16')
    model = nn.DataParallel(model).cuda()
    print('======Load Model======')
    model.load_state_dict(torch.load(args.model_path))
    print('Loaded successfully!')
    # print('======Test Model======')
    # evaluate(model, device, test_loader)

    if args.target_type == 0:
        for i in range(args.target_count):
            print('=======Attacking Target ' + str(i) + '=======')
            # load targets
            img = transform_test(Image.open('./targets/target_' + str(i) + '.png'))
            # print(img.size())

            # start attack with black-box method
            ini_label = model(img.unsqueeze(0)).max(1, keepdim=True)[1].item()
            target_label = torch.tensor([(ini_label + 1) % 10]).to(device)  # 先默认 target label 为 ini_target + 1
            print('Trying:', ini_label, '------>', target_label.item())
            mask, pattern = optimize_blackbox(model, img.unsqueeze(0), target_label)

            mask = mask.detach().permute(1,2,0).squeeze_().cpu().numpy()
            pattern = pattern.detach().permute(1,2,0).cpu().numpy()
            print('mask:', mask.shape, np.min(mask), np.max(mask))
            print('pattern:', pattern.shape, np.min(pattern), np.max(pattern))

            # store mask, fusion, pattern; attacked_target
            # im = Image.fromarray((mask * 255).astype(np.uint8))
            # im.save(pert_path + '/mask_' +  str(i) + '.png')
            # im = Image.fromarray((pattern * 255).astype(np.uint8))
            # im.save(pert_path + '/pattern_' + str(i) + '.png')
            # im = Image.fromarray((mask.reshape([32,32,1]) * pattern * 255).astype(np.uint8))
            # im.save(pert_path + '/fusion_' + str(i) + '.png')

            # img = img.detach().permute(1,2,0).cpu().numpy()
            # im = Image.fromarray(((mask.reshape([32,32,1]) * pattern + img) * 255).astype(np.uint8))
            # im.save(adv_path + '/adv_' + str(i) + '.png')


if __name__ == '__main__':
    main()