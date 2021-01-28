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
np.set_printoptions(threshold=sys.maxsize)

# Hyperparameters
in_channel = 3
num_classes = 1
learning_rate = 1e-3
batch_size = 1
num_epochs = 5
num_data = 10 # Keep this a multiple of 10


csvfile = "traindatainfo.csv" # The file that contains image and binary ground truth as row elements

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
    def __init__(self, csvfile, newW, newH, transform=None):
        """
            Take the csv file, dimensions for resizing image and transformations as inputs arguments
        """
        self.csvfile = csvfile
        self.newH = newH
        self.newW = newW
        self.transform = transform
        with open(self.csvfile, 'r') as f:
            reader = csv.reader(f)
            self.data = list(reader)[:num_data] # Only a small subset of data is used for now

    def __len__(self):        
        return len(self.data) # Returns the length of entire dataset

    @classmethod
    def preprocess(cls, img, gt, newW, newH):
        
        # Read image and normalize it
        img = cv2.resize(img, (newW, newH))
        norm_image = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # Read ground truth and normalize it
        gt = cv2.resize(gt, (newW, newH))
        norm_gt = cv2.normalize(gt, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        return(norm_image, norm_gt)


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
            
        return (img, binary_gt)

# Use gpu if available as device else use cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Load Data
dataset = LaneDataset(
    csvfile=csvfile,
    newW = 512,
    newH = 256,
    transform = transforms.ToTensor(),
)


train_set, test_set = torch.utils.data.random_split(dataset, [int(num_data*0.5), int(num_data*0.5)]) # The train and test should add up to total 
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Model
model = Unet(in_channel, num_classes) # Import the model

# pring the number of model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Number of trainable parameters in the model: ", count_parameters(model))


model.to(device)

# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss() # Used for binary class

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

one_gt = None
# Train Network

print("Network training...")
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch, num_epochs - 1))
	losses = []
	for batch_idx, (data, targets) in enumerate(train_loader):
		# Get data to cuda if possible
		data = data.to(device=device)
		targets = targets.to(device=device)
		one_gt = targets

		# forward
		logits = model(data)
		loss = criterion(logits, targets)
		losses.append(loss.item())
		# backward
		optimizer.zero_grad()
		loss.backward()
		# gradient descent or adam step
		optimizer.step()
	if epoch%5 == 4:
        torch.save(model, './saved_models/saved_model-epoch{}.pth'.format(epoch+1))
        print("Model Saved!!!!")
    print(f"Cost after epoch {epoch} is {sum(losses)/len(losses)}")

def get_dice_loss(x, y):
    eps = 0.1
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

            # print("Here is the predictions done: ", scores.numpy())
            op_img = np.zeros((scores.numpy().shape[2:]))
            print("op_image size: ", op_img.shape)
            
            op_img = np.where(scores.numpy()>0.3, 255, op_img)
            # op_img = Image.fromarray(op_img)
            # op_img.show()
            print(op_img)
            cv2.imshow('alfjk',op_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()  

            # scores = one_gt
           
            

            
            # y = y.view(-1)
            # scores = scores.view(-1)
            
            # intersection = (scores * y).sum()                            
            # dice = (2*intersection + smooth)/(scores.sum() + y.sum() + smooth)  
            loss_acc.append(get_dice_loss(scores, y))


        
        print("Accuracy is: ", sum(loss_acc)/batch_size)

        
    model.train()


print("Checking accuracy on Training Set")
# check_accuracy(train_loader, model)

# print("Checking accuracy on Test Set")
# check_accuracy(train_loader, model)

"""
config 1:

len(dataset): 20 (15, 5)
batch size = 1
epochs = 5

Cost at epoch 0 is 0.5180770417054494
Cost at epoch 1 is 0.3185486614704132
Cost at epoch 2 is 0.24270479778448742
Cost at epoch 3 is 0.19452664852142335
Cost at epoch 4 is 0.16116810739040374

final train accuracy = 0.0432

config 2:
len(dataset): 20 (15,5)
batch size = 1
epochs = 10

Cost at epoch 0 is 0.461492113272349
Cost at epoch 1 is 0.2873123188813527
Cost at epoch 2 is 0.21936607360839844
Cost at epoch 3 is 0.17209940850734712
Cost at epoch 4 is 0.1399588038523992
Cost at epoch 5 is 0.11982663869857788
Cost at epoch 6 is 0.10673108249902725
Cost at epoch 7 is 0.0975310742855072
Cost at epoch 8 is 0.09037074943383534
Cost at epoch 9 is 0.08506367554267248
Checking accuracy on Training Set

Accuracy is:  0.08264


"""
    