<<<<<<< HEAD
import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default='../datasets/imagenet1000',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1,
                                 help='input batch size')
        self.parser.add_argument('--max_epsilon', default=16., type=float)
        self.parser.add_argument('--iter', default=100, type=int)
        self.parser.add_argument('--confidence', default=0, type=float)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--name', type=str, default='greedyfool')
        self.parser.add_argument('--label_filepath', type=str, default='../datasets/imagenet_label.txt')
    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        self.opt = opt
        return self.opt

=======
import argparse
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', default='../datasets/imagenet1000',
                                 help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1,
                                 help='input batch size')
        self.parser.add_argument('--max_epsilon', default=16., type=float)
        self.parser.add_argument('--iter', default=100, type=int)
        self.parser.add_argument('--confidence', default=0, type=float)
        self.parser.add_argument('--phase', type=str, default='test')
        self.parser.add_argument('--name', type=str, default='greedyfool')
        self.parser.add_argument('--label_filepath', type=str, default='../datasets/imagenet_label.txt')
    def parse(self):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()
        self.opt = opt
        return self.opt

>>>>>>> 0b9e7e3357fc6994362217d2b32507c34a28e0f3
