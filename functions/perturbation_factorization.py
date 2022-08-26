from torchvision import transforms

from functions.pf_utils.utils import *
from functions.pf_utils.main import batch_train


def perturbation_attack_black(target_model, dataloader, config, **kwargs):
    # freeze model parameters
    for param in target_model.parameters():
        param.requires_grad = False
    
    for idx, data in enumerate(dataloader):
        if config.dataset_type == 'ImageNet':
            image, label, name = data
        elif config.dataset_type == 'Cifar10':
            image, label = data
            name = idx
        image = image.cpu().clone()
        image = image.squeeze(0)
        image = transforms.ToPILImage()(image)
        batch_train(target_model, image, name)
