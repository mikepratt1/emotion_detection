from torchvision import transforms
import torchvision
import torch
from collections import Counter
import pandas as pd 

def create_dataloaders(train_dir, test_dir, transform, batch_size, device):

    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                    transform=transform)
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                    transform=transform)
    
    # # Create the sampler to deal with class imbalances 
    train_distribution = dict(Counter(train_dataset.targets))
    class_weight = torch.Tensor([len(train_dataset) / c for c in pd.Series(train_distribution).sort_index()])

    # Get all labels in the training dataset
    labels = torch.tensor(train_dataset.targets)

    # Compute the weights for all labels in one go
    sample_weights = class_weight[labels]

    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    print("[INFO] Sampler created")
    
    # Get class names
    class_names = train_dataset.classes

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              sampler=sampler)
    
    return train_dataloader, test_dataloader, class_names
