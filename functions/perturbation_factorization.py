from torchvision import transforms

from functions.pf_utils.flags import parse_handle
from functions.pf_utils.utils import *
from functions.pf_utils.main import batch_train


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.suc = 0
        self.asr = 0
        self.suc_sum = 0

        self.l0 = 0
        self.l0_avg = 0
        self.l0_sum = 0

        self.l1 = 0
        self.l1_avg = 0
        self.l1_sum = 0

        self.l2 = 0
        self.l2_avg = 0
        self.l2_sum = 0

        self.linf = 0
        self.linf_avg = 0
        self.linf_sum = 0
        
        self.count = 0

    def update(self, suc, l0, l1, l2, linf, n=1):
        self.count += n

        self.suc_sum += suc * n
        self.asr = self.suc_sum / self.count

        self.l0_sum += l0 * n
        self.l0_avg = self.l0_sum / self.count

        self.l1_sum += l1 * n
        self.l1_avg = self.l1_sum / self.count

        self.l2_sum += l2 * n
        self.l2_avg = self.l2_sum / self.count

        self.linf_sum += linf * n
        self.linf_avg = self.linf_sum / self.count

        


def my_arg_parse(config):
    catg = '1000' if config.dataset_type == 'ImageNet' else '10'
    lambda1 = '1e-2' if config.dataset_type == 'ImageNet' else '1e-3'
    parser = parse_handle()
    return parser.parse_args(['--img_resized_width', str(config.image_size),
                              '--img_resized_height', str(config.image_size),
                              '--init_lambda1', lambda1,
                              '--categories', catg,
                              '--maxIter_e', str(config.maxIter_e), 
                              '--maxIter_g', str(config.maxIter_g)])


def perturbation_attack_black(target_model, dataloader, config, **kwargs):
    # Average meter
    result_avg = AverageMeter()
    #parsing input parameters
    args = my_arg_parse(config)

    # freeze model parameters
    for param in target_model.parameters():
        param.requires_grad = False
    
    for idx, data in enumerate(dataloader):
        if idx == 100:
            break
        if config.dataset_type == 'ImageNet':
            image, label, name = data
            name = name[0]
        elif config.dataset_type == 'Cifar10':
            image, label = data
            name = str(idx)
        image = image.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        result = batch_train(target_model, image, name, args, config)
        result_avg.update(*result)
        
        print("//////Overall statistics://////")
        print('statistic information: success-attack-image/total-attack-image= %d/%d, attack-success-rate=%f, L0=%f, L1=%f, L2=%f, L-inf=%f \n' \
            %(result_avg.suc_sum, result_avg.count, result_avg.asr, result_avg.l0_avg, result_avg.l1_avg, result_avg.l2_avg, result_avg.linf_avg))
        
        if idx == 99:
            break
