import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(data_dir, batch_size=32):
    print("Initializing image transformations...")
    
    # Neural networks expect images to be the exact same size and normalized
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(), # Data augmentation prevents overfitting
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Standard ImageNet metrics
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading image datasets from {data_dir}...")
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    
    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
        for x in ['train', 'val']
    }
    
    class_names = image_datasets['train'].classes
    print(f"Detected classes: {class_names}")
    
    return dataloaders, class_names
