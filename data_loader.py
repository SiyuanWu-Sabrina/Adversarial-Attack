import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from imagenet_dataset import ImageNetDataset


def data_factory(dataset_type = 'ImageNet', no_cuda = False, batch_size = 1):
    use_cuda = not no_cuda and torch.cuda.is_available()
    if dataset_type == 'Cifar10':
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        dataset = torchvision.datasets.CIFAR10(
            root='./datasets', train=False, download=True, 
            transform=transforms.Compose([
                transforms.CenterCrop(31),
                transforms.ToTensor()
            ])
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, **kwargs)
    elif dataset_type == 'ImageNet':
        im_size = 299
        mean_arr = (0.5, 0.5, 0.5)
        stddev_arr = (0.5, 0.5, 0.5)
        dataset = ImageNetDataset(
            image_dir='./datasets/imagenet1000',
            label_filepath="./datasets/imagenet_label.txt",
            transform=transforms.Compose([
                transforms.CenterCrop(im_size),
                transforms.ToTensor(),
                transforms.Normalize(mean_arr, stddev_arr)
            ]),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)
 