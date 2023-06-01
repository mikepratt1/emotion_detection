from torchvision import transforms
import torchvision
import torch
from collections import Counter
import pandas as pd 

def create_dataloaders(train_dir, test_dir, transform, batch_size):

    test_dataset = torchvision.datasets.ImageFolder(root=test_dir,
                                                    transform=transform)
    train_dataset = torchvision.datasets.ImageFolder(root=train_dir,
                                                    transform=transform)
    
    # Create the sampler to deal with class imbalances 
    train_distribution = dict(Counter(train_dataset.targets))
    class_weight = torch.Tensor([len(train_dataset) / c for c in pd.Series(train_distribution).sort_index()])

    sample_weight = [0]*len(train_dataset)
    for idx, (img, label) in enumerate(train_dataset):
        weight = class_weight[label]
        sample_weight[idx] = weight

    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weight, num_samples=len(train_dataset), replacement=True)

    # Get class names
    class_names = train_dataset.classes

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=batch_size,
                                              sampler=sampler)
    
    return train_dataloader, test_dataloader, class_names
