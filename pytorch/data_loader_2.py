import torchvision
import torch

def load_datasets(batch_size, transform=None):
    train_path = '../data/train/Training'
    val_path = '../data/train/Validation'

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    val_dataset = torchvision.datasets.ImageFolder(
        root=val_path,
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )

    return train_loader, val_loader

def test_dataset(batch_size, transform=None):
    test_path = '../data/train/Testing'
    test_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return test_loader
