import os
import sys
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from PIL import Image
import multiprocessing
# --- GLOBAL HELPER FUNCTIONS (Must be outside to work on Windows) ---
def to_rgb(x):
    return x.convert('RGB')

def get_dataloaders(config):
    # Detect Environment for Workers
    # Linux/Colab (posix) -> Fork -> Fast (2+ workers)
    # Windows (nt) -> Spawn -> Fails in Notebooks -> Slow (0 workers)
    if os.name == 'nt':
        workers = min(multiprocessing.cpu_count(),4)
        print(f"Windows detected: Using {workers}")
    else:
        workers = 2
        print(f"Linux/Colab detected: Using {workers} workers for speed.")

    img_size = config['img_size']

    # 1. Train Transforms (Global function used instead of lambda)
    train_tf = transforms.Compose([
        transforms.Lambda(to_rgb), 
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=45),
        transforms.RandomPerspective(distortion_scale=0.4, p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=15),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    # 2. Val Transforms
    val_tf = transforms.Compose([
        transforms.Lambda(to_rgb),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. Load & Split
    full_train = datasets.ImageFolder(config['data_dir'], transform=train_tf)
    full_val   = datasets.ImageFolder(config['data_dir'], transform=val_tf)

    ds_len = len(full_train)
    train_sz = int(0.8 * ds_len)
    val_sz = ds_len - train_sz

    gen = torch.Generator().manual_seed(42)
    train_ds, _ = random_split(full_train, [train_sz, val_sz], generator=gen)
    _, val_ds   = random_split(full_val,   [train_sz, val_sz], generator=gen)

    # 4. Create Loaders with Dynamic Workers
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, 
                          num_workers=workers, pin_memory=True)
    val_dl   = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, 
                          num_workers=workers, pin_memory=True)

    return train_dl, val_dl, full_train.classes
