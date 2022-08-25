import os

from torch.utils.data import Dataset
from PIL import Image


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class ImageNetDataset(Dataset):
    def __init__(self, image_dir, label_filepath, transform=None):
        """
            Load metadata of the imagenet dataset, including image-label mapping,
        image file path, and corresponding image names.
        """
        with open(label_filepath, 'r') as fp:
            data = [i.strip() for i in fp.readlines()]
            name = [i.split(' ')[0] for i in data]
            label = [int(i.split(' ')[1]) for i in data]
        mappings = dict(zip(name, label))

        images = os.listdir(image_dir)
        self.all_paths = []
        self.all_labels = []
        self.all_names = []
        for img in images:
            image_path = os.path.join(image_dir, img)
            self.all_paths.append(image_path)
            self.all_labels.append(mappings[img])
            self.all_names.append(img.split('.')[0])

        self.size = len(self.all_paths)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        path = self.all_paths[index % self.size]
        data = load_img(path)
        if self.transform is not None:
            data = self.transform(data)
        label = self.all_labels[index % self.size]
        # name = self.all_names[index % self.size]
        # return {'image': data, 'label': label, 'name': name}
        return (data, label)
