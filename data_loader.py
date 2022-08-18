import os

from torch.utils.data import Dataset
from PIL import Image


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img


class ImageNetDataset(Dataset):
    def __init__(self, image_dir, label_filepath, transform=None):
        self.transform = transform
        self.image_dir = image_dir

        with open(label_filepath, 'r') as fp:
            data = [i.strip() for i in fp.readlines()]
            name = [i.split(' ')[0] for i in data]
            label = [int(i.split(' ')[1]) for i in data]
        assert len(name) == len(label)
        mappings = dict(zip(name, label))

        imgs = os.listdir(self.image_dir)
        self.A_paths = []
        self.A_labels = []
        for img in imgs:
            imgpath = os.path.join(self.image_dir, img)
            self.A_paths.append(imgpath)
            self.A_labels.append(mappings[img])
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        # print("read meta done")
        self.initialized = False

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)

        A_label = self.A_labels[index % self.A_size]
        return {'A': A, 'label': A_label}
