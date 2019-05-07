import torchvision

def load_datasets(batch_size, transform=None):
    train_path = '../data/train/Training'
    val_path = '../data/train/Validation'

    train_dataset = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=torchvision.transforms.ToTensor()
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