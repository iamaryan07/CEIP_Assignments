"""Shared torchvision transform pipelines."""
from torchvision import transforms

from . import config


def baseline_transforms(train: bool):
    """No ImageNet normalization -- the baseline CNN is trained from scratch."""
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])
    return transforms.Compose([transforms.ToTensor()])


def resnet_transforms(train: bool, image_size: int = 224):
    """Resize to the ImageNet-pretrained input size and normalize with the
    ImageNet mean/std the backbone was originally trained with."""
    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ]
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    return transforms.Compose(ops)
