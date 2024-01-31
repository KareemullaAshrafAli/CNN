import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as datasets 







def Get_DataLoader_Objects(args):

    train_path = 'Dataset/'
    test_path = 'Datasets/'


    if (args.dataset == 'cifar10'):
        args.dataset = 'CIFAR10'
        train_path += 'CIFAR10/CIFAR10_Train_Set'
        test_path +=  'CIFAR10/CIFAR10_Test_Set'


    elif (args.dataset == 'cifar100'):
        args.dataset = 'CIFAR100'
        train_path += 'CIFAR100/CIFAR100_Train_Set'
        test_path +=  'CIFAR100/CIFAR100_Test_Set'


    elif (args.dataset == 'emnist'):
        args.dataset = 'EMNIST'
        train_path += 'EMNIST/EMNIST_Train_Set'
        test_path +=  'EMNIST/EMNIST_Test_Set'


    elif (args.dataset == 'fashionmnist'):
        args.dataset = 'FashionMNIST'
        train_path += 'FashionMNIST/FashionMNIST_Train_Set'
        test_path +=  'FashionMNIST/FashionMNIST_Test_Set'



    # Transformation for the Images
        
    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])



    
    dataset_class = getattr(datasets, args.dataset)
    train_data = dataset_class(root = train_path, download = True, train = True, transform = transformation)
    test_data = dataset_class(root = test_path, download = True, train = False, transform = transformation)




    # Create the DataLoaders

    train_loader = DataLoader(train_data, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, drop_last = True)
    test_loader = DataLoader(test_data, shuffle = True, batch_size = args.batch_size, num_workers = args.num_workers, drop_last = True) 



    return train_loader, test_loader