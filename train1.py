from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import torch.nn as nn
from torch import optim
import Unet_model
from Unet_model import Unet
from torch.utils.data import DataLoader, random_split, Dataset
import csv
import cv2
import sys
from parameters import Parameters
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)



# dir_checkpoint = "./chkpt/"

# Define the class for pytorch Dataset package
class LaneDataset(Dataset):
    """
        The class for pytorch dataloader should contain the following three functionalities:
        a) init: to describe and initialize parameters
        b) __len__: to return the length of the dataset used
        c) __getitem__: to return elements from the dataset

        Any addditional function can be used for example for image preprocessing.
        The use of this class makes data loading very convinient and simplified. The dataloader can later be used through this class
        to generate batches, splitting training, testing and validation data.
    """
    def __init__(self, csvfile, newW, newH, num_data, transform=None):
        """
            Take the csv file, dimensions for resizing image and transformations as inputs arguments
        """
        self.csvfile = csvfile
        self.newH = newH
        self.newW = newW
        self.transform = transform
        with open(self.csvfile, 'r') as f:
            reader = csv.reader(f)
            self.data = list(reader)[51:53] # Only a small subt of data is used for now

    def __len__(self):        
        return len(self.data) # Returns the length of entire dataset

    @classmethod
    def preprocess(cls, img, gt, newW, newH):
        
        # Read image and normalize it
        img = cv2.resize(img, (newW, newH))

        # plt.imshow(img)
        # plt.show()
        # exit()
        # img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Read ground truth and normalize it
        gt = cv2.resize(gt, (newW, newH))        

        # obtain a two channel gt using single channel gt
        gt0 = np.zeros((newW, newH))
        gt1 = np.zeros((newW, newH))

        gt0 = np.where(gt == 0, 1, 0)  # Rest of the world
        
        gt1 = 1-gt0                     # Lane lines
        
        final_gt = np.stack((gt0, gt1), axis = -1)

        # Uncomment following line to see the ground truth images
        # plt.imshow(cv2.split(final_gt)[0], cmap = plt.cm.gray)
        # plt.show()
        # exit()

        
        return(img, final_gt)


    def __getitem__(self, i):
        """
            The function takes the index and returns the corresponding data element.
            How to retrieve a data element using this index is implemented in this function.
        """

        # Read original image and ground truth
        img_path = self.data[i][0]
        binary_gt_path = self.data[i][1]        
        img = cv2.imread(img_path)
        binary_gt = cv2.imread(binary_gt_path, 0)        
        
        # Get processed data 
        img, binary_gt = self.preprocess(img, binary_gt, self.newW, self.newH)        

        if self.transform: 
            img = self.transform(img)
            binary_gt = self.transform(binary_gt)
            
        return (img, binary_gt.float())



# pring the number of model parameters

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# print("Number of trainable parameters in the model: ", count_parameters(model))


    
def get_dice_loss(x, y):
    """
        let x = [0 0 0 1 0 1 0 0 0 1 1 0 1] sum = 5
        let y = [0 1 1 0 1 0 1 1 0 0 1 1 1] sum = 8
        x . y = [0 0 0 0 0 0 0 0 0 0 1 0 1] sum = 2
        op = 2/13

    """

    eps = 0.1
    assert x.size() == y.size(), "sizes not equal in the dice loss function"

    inter = torch.dot(x.view(-1), y.view(-1))
    union = torch.sum(x) + torch.sum(y) + eps

    return((2.*inter + eps)/ union)

# Check accuracy on training to see how good our model is
def check_accuracy(loader, model):
    
    model.eval()

    with torch.no_grad():
        loss_acc = []
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)    

            scores = model(x)

            loss_acc.append(get_dice_loss(scores, y))
        
        print("Accuracy is: ", sum(loss_acc)/batch_size)

        
    model.train()



# check_accuracy(train_loader, model)

# print("Checking accuracy on Test Set")
# check_accuracy(train_loader, model)

if __name__ == "__main__":
    csvfile = "traindatainfo.csv" # The file that contains image and binary ground truth as row elements
    p = Parameters()
    # Hyperparameters
    in_channel = p.in_channel
    num_classes = p.num_classes
    learning_rate = p.learning_rate
    batch_size = p.batch_size
    num_epochs = p.num_epochs
    num_data = p.num_data
    train_fraction = p.train_fraction
    test_fraction = 1 - train_fraction

    # Use gpu if available as device else use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Data
    dataset = LaneDataset(
        csvfile=csvfile,
        newW = 512,
        newH = 256,
        transform = transforms.ToTensor(),
        num_data = num_data
    )


    train_set, test_set = torch.utils.data.random_split(dataset, [int(num_data*train_fraction), int(num_data*test_fraction)]) # The train and test should add up to total 
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    # Model
    model = Unet(in_channel, num_classes) # Import the model

    model.to(device)
    # Loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss(reduce = False) # Used for binary class    

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    one_gt = None
    # Train Network

    print("Network training...")
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        losses = []
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Get data to cuda if possible
            data = data.to(device=device)
            targets = targets.to(device=device)

            # forward
            logits = model(data)

            weight = torch.tensor([0.01, 0.99])
            weight_ = weight[targets.data.view(-1).long()].view_as(targets)

            # print(weight_.size())

            # exit()
            
            loss = criterion(logits, targets)
            loss_class_weighted = loss * weight_
            loss_class_weighted = loss_class_weighted.mean()



            # loss = criterion(logits, targets)
            # losses.append(loss.item())
            losses.append(loss_class_weighted.item())
            # backward
            optimizer.zero_grad()
            loss_class_weighted.backward()
            # gradient descent or adam step
            optimizer.step()
        if epoch%10 == 9:
            torch.save(model, './saved_models/saved_model-epoch{}.pth'.format(epoch+1))
            print("Model Saved!!!!")
            print("Checking accuracy on Training Set")
            check_accuracy(train_loader, model)
        print(f"Cost after epoch {epoch+1} is {sum(losses)/len(losses)}")

    print("Checking accuracy on Training Set")
    check_accuracy(train_loader, model)

