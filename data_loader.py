from torch.utils.data import DataLoader, random_split
from torchvision.models import ViT_B_32_Weights
from torchvision.datasets import CIFAR10
import torch

def load_data(batch_size: int = 128,
              shuffle: bool = True,
              drop_last: bool = True,
              num_workers: int = 4,
              root: str = './data',
              val_split: float = 0,
              custom_train_transform = None,
              custom_test_transform = None):
    """Load CIFAR-10 dataset with optional preprocessing.

    Parameters:
        batch_size (int): Batch size for Data_loader.
        shuffle (bool): Whether to shuffle the data or not.
        drop_last (bool): Whether to ignore last batch or not.
        num_workers (int): Number of processors loading batches.
        root (str): Root directory for data.
        val_split (float): Percentage of training data to use as validation.
        custom_train_transform: Custom transformations applied to training set.
        custom_test_transform: Custom transformations applied to testing set.
    Returns:
        transform (ImageClassification): Dataset transformations.
        train_loader (Data_loader): Training data.
        test_loader (Data_loader): Test data.
        val_loader (Data_loader): Validation data (optional).
    """
    if custom_train_transform is not None:
        train_transform = custom_train_transform
    else:
        train_transform = ViT_B_32_Weights.IMAGENET1K_V1.transforms()

    if custom_test_transform is not None:
        test_transform = custom_test_transform
    else:
         test_transform = ViT_B_32_Weights.IMAGENET1K_V1.transforms()

    train_set = CIFAR10(root=root, train=True, download=True,
                        transform=train_transform)
    test_set = CIFAR10(root=root, train=False, download=True,
                       transform=test_transform)

    # train/val split if applicable
    if val_split > 0:
        val_size = int(len(train_set) * val_split)
        train_size = len(train_set) - val_size
        train_set, valset = random_split(train_set, [train_size, val_size])
        val_loader = DataLoader(valset, batch_size=batch_size,
                                shuffle=False, drop_last=False,
                                num_workers=num_workers)
    else:
        val_loader = None

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=shuffle, drop_last=drop_last,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers)

    if torch.cuda.is_available():
        for data in [train_loader, val_loader, test_loader]:
            if data is not None:
                for images, labels in data:
                    images, labels = images.to('cuda'), labels.to('cuda')

    return train_loader, test_loader, val_loader