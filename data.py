from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib
import tarfile
from torchvision.transforms import Compose, ToTensor

from dataset import DatasetFromFolderTest, DatasetFromFolder

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(gt_dir, seq_dir, nFrames, data_augmentation, file_list, patch_size, ds, debug):
    return DatasetFromFolder(gt_dir, seq_dir, nFrames, data_augmentation, file_list, patch_size, ds, debug,
                             transform=transform())


def get_eval_set(gt_dir, seq_dir, nFrames, data_augmentation, file_list, patch_size, ds, debug):
    return DatasetFromFolder(gt_dir, seq_dir, nFrames,  data_augmentation, file_list, patch_size, ds, debug,
                             transform=transform())

def get_test_set(gt_dir, seq_dir, nFrames, file_list):
    return DatasetFromFolderTest(gt_dir, seq_dir, nFrames, file_list, transform=transform())

