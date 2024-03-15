from utils import *
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw
from IPython.display import display
from matplotlib import pyplot as plt
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor, pil_to_tensor
random.seed(1)
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from torch import nn, optim


def parse_images_from_folder(image_dir_path, xml_dir_path,pixel_dir_path):
    xml_files = os.listdir(xml_dir_path)
    image_files = os.listdir(image_dir_path)
    pixel_files = os.listdir(pixel_dir_path)
    output = []
    for i in range(len(xml_files)):
        img_file_path = image_dir_path + "/" + image_files[i]
        xml_file_path = xml_dir_path + "/" + xml_files[i]
        pixel_gt_file_path = pixel_dir_path + "/" + pixel_files[i]
        page = Page.from_file(img_file_path,xml_file_path,pixel_gt_file_path)
        output.append(page)
    return output

haha = parse_images_from_folder("all/img-CS18/img/training","all/PAGE-gt-CS18/PAGE-gt/training","all/pixel-level-gt-CS18/pixel-level-gt/training")

class PatchesDataset(Dataset):
    def get_crops(self):
        for page in self.pages:
            for patch in page.grid:
                self.base_patches.append(patch.img)
                self.base_labels.append(patch.gt)
                self.coords.append((patch.x,patch.y,page.name))

    def __init__(self,pages, transform = None ):
        self.pages = pages
        self.transform = transform
        self.base_patches = []
        self.base_labels = []


        self.coords = []
        self.get_crops()
        self.patches = self.base_patches
        self.labels = self.base_labels

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, index):
        patch = Image.fromarray(self.patches[index].astype('uint8'),'RGB')
        label = Image.fromarray(self.labels[index].astype('uint8'),'L')

        patch = self.transform(patch)
        label = pil_to_tensor(label).squeeze().long()
        return patch,label

    def random_patch_generator(self,num_patches):
        """
        Divides an image represented by a numpy array into square patches.

        Parameters:
        - patch_size: integer, the size of each square patch in pixels.

        Returns:
        - A list of Patches, each representing a square patch of the original image.
        """
        images = []
        labels = []
        patch_size = self.base_patches[0].shape[0]
        height, width, _ = self.pages[0].img.shape
        for page in self.pages:
            for i in range(num_patches):
                start_x = random.randint(0,width-832)
                start_y = random.randint(0,height-832)
                end_x = start_x + patch_size
                end_y = start_y + patch_size

                content = page.img[start_y:end_y,start_x:end_x, :]
                gt = page.gt[start_y:end_y,start_x:end_x]
                labels.append(gt)
                images.append(content)
        self.patches = self.base_patches + images
        self.labels = self.base_labels + labels

patches_transforms = transforms.Compose([
    transforms.Resize((832, 832)),
    transforms.ToTensor(),
])

num_classes = 4  # For example, 21 for the VOC dataset
batch_size = 2
num_epochs = 1
learning_rate = 1e-3

# Dataset and DataLoader
train_dataset = PatchesDataset(haha[:3],patches_transforms)
train_dataset.random_patch_generator(0)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
model = deeplabv3_resnet50(pretrained=False, num_classes=num_classes)
model.train()

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        print(images.shape,labels.shape)
        # Forward pass
        outputs = model(images)
        print("Hello")
        loss = criterion(outputs['out'], labels)
        print(loss)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print('Training finished!')
