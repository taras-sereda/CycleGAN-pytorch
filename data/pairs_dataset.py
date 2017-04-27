import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data.image_folder import ImageFolder


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B

    def __iter__(self):
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        return self

    def __next__(self):
        A, A_paths = next(self.data_loader_A_iter)
        B, B_paths = next(self.data_loader_B_iter)
        return {'A': A, 'A_paths': A_paths,
                'B': B, 'B_paths': B_paths}


class UnalignedDataLoader(object):
    def __init__(self, params):
        transform = transforms.Compose([
            transforms.Scale(size=(params.load_size, params.load_size)),
            transforms.RandomCrop(size=(params.height, params.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5))
        ])
        dataset_A = torch.utils.data.DataLoader(
            ImageFolder(root=params.data_root + '/' + params.phase + 'A',
                        transform=transform),
            num_workers=params.num_workers,
            shuffle=params.shuffle)

        dataset_B = torch.utils.data.DataLoader(
            ImageFolder(root=params.data_root + '/' + params.phase + 'B',
                        transform=transform),
            num_workers=params.num_workers,
            shuffle=params.shuffle)

        self.dataset_A = dataset_A
        self.dataset_B = dataset_B
        self.paired_data = PairedData(self.dataset_A, self.dataset_B)

    def load_data(self):
        return self.paired_data

    def __len__(self):
        return len(self.dataset_A)
