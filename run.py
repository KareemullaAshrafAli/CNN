# Import Libraries

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse 
import os
from tqdm import tqdm
import time

from dataloader import Get_DataLoader_Objects
from architecture import AlexNet

from CNN_Architecture import AlexNet, LeNet5, VGG16








parser = argparse.ArgumentParser(description = 'CNN Parametric Experiment')


# Device Settings
parser.add_argument('--device', default = 'cpu', type = str, help = 'CPU/GPU for training')


# Model Parameters
parser.add_argument('--architecture', default = 'lenet5', type = str, help = 'Architecture for training')
parser.add_argument('--lr', default = 0.01, type = float, help = 'Learning Rate')


# Dataset
parser.add_argument('--dataset', default = 'emnist', type = str, help = 'Name of the dataset')
parser.add_argument('--batch_size', default = 128, type = int, help = 'Training Batch Size')
parser.add_argument('--num_workers', default = 4, type = int, help = 'Number of data loading workers')


# Load, save, resume checkpoints
parser.add_argument('--resume_checkpoint', type = str, help = 'Path of the checkpoint')
parser.add_argument('--start_epoch', default = 0, type = int, help = 'Starting epoch for training')
parser.add_argument('--end_epoch', default = 10, type = int, help = 'Ending epoch for training')





















def main():

    args = parser.parse_args()


    # Check for GPU availability

    if ((args.device == 'gpu') and (torch.cuda.is_available())):
        args.device = 'cuda:0'
    else:
        args.device = 'cpu'
        print('\n=> GPU cannot be accessed')
        
    print('=> Using {} for training the model'.format(args.device))




    # Create the Model

    net = AlexNet(100, 0.3).to(args.device)
    optimizer = torch.optim.SGD(net.parameters(), lr = args.lr)
    lossfun = nn.CrossEntropyLoss()



    #  Check for checkpoints

    if (args.resume_checkpoint is not None):
        if (os.path.isfile(args.resume_checkpoint)):
            print("=> Loading checkpoint: {}".format(args.resume_checkpoint))
            checkpoint_path = args.resume_checkpoint
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            args.start_epoch = checkpoint['start_epoch']
            print("=> Checkpoint {} successfully loaded. Start epoch: {}".format(args.resume_checkpoint, args.start_epoch))

        else:
            print("No checkpoint found at {}".format(args.resume_checkpoint))



    # Call the main worker method
            
    main_worker(args, net, optimizer, lossfun)    














def main_worker(args, net, optimizer, lossfun):
    
    # Model, Optimizer, Loss Function ready

    # Import the Data Loaders
    train_loader, test_loader = Get_DataLoader_Objects(args)


    
    # Train the Model

    start_time = time.time()
    epoch_range = range(args.start_epoch, args.end_epoch, 1)
    train_acc = []
    train_loss = []


    for epochi in tqdm(epoch_range, desc = '\nTraining Progress: '):
        net.train()
        train_batch_acc = []
        train_batch_loss = []

        for _, batch in enumerate(train_loader):
            mini_train_batch = batch[0].to(args.device)
            mini_label_batch = batch[1].to(args.device)

            yhat = net(mini_train_batch).to(args.device)
            loss = lossfun(yhat, mini_label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchAcc = torch.mean((torch.argmax(yhat, axis = 1) == mini_label_batch).float())
            train_batch_acc.append(batchAcc.item())

            train_batch_loss.append(loss.item())

        train_acc.append(np.mean(train_batch_acc))
        train_loss.append(np.mean(train_batch_loss))

       

    total_time = time.time() - start_time
    print("\n=> Took {} sec to complete training".format(total_time))





    # Test the Model
    
    net.eval()
    test_batch_acc = []

    with torch.no_grad():
        for _, test_batch in enumerate(test_loader):
            mini_test = test_batch[0].to(args.device)
            y = test_batch[1].to(args.device)

            yhat = net(mini_test).to(args.device)
            pred = torch.argmax(yhat, axis = 1).item()
            acc = 100*torch.mean((pred == y).float())
            test_batch_acc.append(acc.item())

    print("\nTest Set Accuracy = {}".format(np.mean(test_batch_acc)))





    # Plot the results

    fig, ax = plt.subplots(1, 2, figsize = (10,6))

    ax[0].plot(train_acc)
    ax[1].plot(train_loss)



# Call main method
    
if __name__ == '__main__':
    main()