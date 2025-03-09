import pandas as pd
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Load image paths and attributes
data_dir = '../dataset/img_align_celeba/'
attr_file = '..//dataset/list_attr_celeba.csv'

# Load attributes
attributes = pd.read_csv(attr_file)
# Here, you can filter or process attributes as needed

# Create your dataset class
class CelebADataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.attr = attributes  # Load attributes here

    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        # You can also return the attributes
        return img, self.attr.iloc[index]

# Create DataLoader
dataset = CelebADataset(data_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Example of iterating through the DataLoader
for images, attrs in data_loader:
    # Your training code here
    pass