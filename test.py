import torch
import cv2
from torch.utils.data import DataLoader, random_split, Dataset
from train1 import LaneDataset
import torchvision.transforms as transforms
import csv
import Unet_model
from Unet_model import Unet
import matplotlib.pyplot as plt
import numpy as np
from parameters import Parameters
from PIL import Image

def draw_figures(img, gt, op):
    
    img = torch.squeeze(img)
    gt = torch.squeeze(gt)


    trans = transforms.ToPILImage()

    fig, (ax1, ax2, ax3) = plt.subplots(3)
    
    ax1.imshow(trans(img))
    ax2.imshow(trans(gt), cmap = plt.cm.gray)
    ax3.imshow(trans(op), cmap = plt.cm.gray)

    plt.show()
    
    
    
    return()





def check_accuracy(loader, model):    
    model.eval()
    with torch.no_grad():
        loss_acc = []
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)    

            scores = model(x)
            
            op = torch.argmax(scores, dim = 1)
            loss_acc.append(get_dice_loss(scores, y))
            draw_figures(x, y, op.float())            
        print("Accuracy is: ", sum(loss_acc)/len(loss_acc))


def get_dice_loss(x, y):
    """
        X, y need to be tensors
        Returns the measure of similarity or dissimilarity of the two tensors.
    """
    eps = 0.1
    assert x.size() == y.size(), "sizes not equal in the dice loss function"
    
    inter = torch.dot(x.view(-1), y.view(-1))
    union = torch.sum(x) + torch.sum(y) + eps

    return((2.*inter + eps)/ union)

p = Parameters()

num_data = p.num_data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csvfile = "traindatainfo.csv"

dataset = LaneDataset(
    csvfile=csvfile,
    newW = 512,
    newH = 256,
    transform = transforms.ToTensor(),
    num_data = num_data
)

batch_size = 1
train_set, test_set = torch.utils.data.random_split(dataset, [int(num_data*0.5), int(num_data*0.5)]) # The train and test should add up to total 
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

path = "saved_models/saved_model-epoch100.pth"
# model = Unet(3, 1)
model = torch.load(path)
# model.eval()

# model = torch.load("./saved_models/saved_model-epoch30.pth")

model.to(device)


print("Checking accuracy Set")
check_accuracy(train_loader, model)

