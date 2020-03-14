from torchvision import datasets, transforms
from models import resnet18

# Note that we don't use cifar10 specific normalization, so generally use 0.5 as mean and std.
mean = 0.5
std = 0.5

clip_min = -1.
clip_max = 1.


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_dataset(data_name='cifar10', data_dir='data', train=True, crop_flip=True):
    """
    Get a dataset.
    :param data_name: str, name of dataset.
    :param data_dir: str, base directory of data.
    :param train: bool, return train set if True, or test set if False.
    :param crop_flip: bool, whether use crop_flip as data augmentation.
    :return: pytorch dataset.
    """

    transform_3d_crop_flip = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([mean] * 3, [std] * 3)
    ])

    transform_3d = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([mean] * 3, [std] * 3)
    ])

    if train:
        # when train is True, we use transform_1d_crop_flip by default unless crop_flip is set to False
        transform = transform_3d if crop_flip is False else transform_3d_crop_flip
    else:
        transform = transform_3d

    if data_name == 'cifar10':
        dataset = datasets.CIFAR10(data_dir, train=train, download=True, transform=transform)
    elif data_name == 'cifar100':
        dataset = datasets.CIFAR100(data_dir, train=train, download=True, transform=transform)
    else:
        raise ('dataset {} is not available'.format(data_name))

    return dataset


def cal_parameters(model):
    """
    Calculate the number of parameters of a Pytorch model.
    :param model: torch.nn.Module
    :return: int, number of parameters.
    """
    return sum([para.numel() for para in model.parameters()])


def get_model(name='resnet18', width=64, n_classes=10):
    classifier = eval(name)(width, n_classes)
    return classifier

