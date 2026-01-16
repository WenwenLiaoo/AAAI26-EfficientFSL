import torch.utils.data as data

from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
import json
import h5py


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, loader=default_loader, embed_path=None):
        self.root = root
        self.imlist = json.load(open(flist, 'r'))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = list(self.imlist[index].items())[0]
        img = self.loader(os.path.join(self.root, "images",  impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.imlist)

from torchvision.datasets import ImageFolder
from utils import transform_train, transform_val
from samplers import CategoriesSampler
from torch.utils.data import DataLoader
def get_data(name, batch_size=64,num_workers=8,test_batch=600, test_way=5, shot=1, query=15):
    root = './dataset/' + name

    val = root+'/val'
    train = root+'/train'
    test = root+'/test'
    train_dataset = ImageFolder(train, transform=transform_train)
    val_dataset = ImageFolder(val, transform=transform_val)    
    test_dataset = ImageFolder(test, transform=transform_val)
    
    train_sampler = CategoriesSampler(train_dataset.targets, batch_size, test_way, shot + query)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler,
                            num_workers=num_workers, pin_memory=True)
    val_sampler = CategoriesSampler(val_dataset.targets, test_batch, test_way, shot + query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler,
                            num_workers=num_workers, pin_memory=True)
    test_sampler = CategoriesSampler(test_dataset.targets, 5*test_batch, test_way, shot + query)
    test_loader = DataLoader(dataset=test_dataset, batch_sampler=test_sampler,
                            num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader