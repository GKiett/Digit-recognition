import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(cfg):
    aug = cfg['data']['augment']
    normalize = transforms.Normalize((0.1307,), (0.3081,)) if cfg['data']['normalization'] == 'zscore' else None

    train_tf = [transforms.RandomAffine(
        degrees=aug['rotate_deg'],
        translate=(aug['translate_px']/28, aug['translate_px']/28),
        scale=(aug['scale_min'], aug['scale_max'])
    ), transforms.ToTensor()]
    if normalize: train_tf.append(normalize)

    test_tf = [transforms.ToTensor()]
    if normalize: test_tf.append(normalize)

    train_ds = datasets.MNIST(cfg['data']['root'], train=True, download=True, transform=transforms.Compose(train_tf))
    test_ds = datasets.MNIST(cfg['data']['root'], train=False, download=True, transform=transforms.Compose(test_tf))

    val_size = int(len(train_ds) * cfg['data']['val_split'])
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size])

    bs, nw = cfg['data']['batch_size'], cfg['data']['num_workers']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, test_loader
