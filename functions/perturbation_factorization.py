from torchvision import transforms

from functions.pf_utils.flags import parse_handle
from functions.pf_utils.utils import *
from functions.pf_utils.main import batch_train


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
    #parsing input parameters
    args = my_arg_parse(config)

    # freeze model parameters
    for param in target_model.parameters():
        param.requires_grad = False
    
    for idx, data in enumerate(dataloader):
        if config.dataset_type == 'ImageNet':
            image, label, name = data
            name = name[0]
        elif config.dataset_type == 'Cifar10':
            image, label = data
            name = str(idx)
        image = image.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        batch_train(target_model, image, name, args, config)
