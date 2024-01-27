import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def Get_DataLoader_Objects(args):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = torchvision.datasets.CIFAR100(root = 'Dataset/Cifar100_Train_Set', download = True, train = True, transform = transform)
    test_data = torchvision.datasets.CIFAR100(root = 'Dataset/Cifar100_Test_Set', download = True, train = False, transform = transform)


    train_loader = DataLoader(train_data, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, drop_last = True)
    test_loader = DataLoader(test_data, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, drop_last = True) 

    return train_loader, test_loader