# import torch.nn.functional as F
import numpy as np
from Unet_parts import *
import torch


class Unet(nn.Module):
	"""
		Create the Unet architecture using the predefined parts
		The structure is as follows:

		INPUT :::Input conv ==> [down1->down2->down3->down4] ==> [up1->up2->up3->up4] ==> Outconvolution::: Output

	"""
	def __init__(self, n_channels, n_classes, bilinear = False):
		super(Unet, self).__init__()
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.bilinear = bilinear

		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)
		factor = 2 if bilinear else 1
		self.down4 = Down(512, 1024 // factor)
		
		self.up1 = Up(1024, 512 // factor, bilinear)
		self.up2 = Up(512, 256 // factor, bilinear)
		self.up3 = Up(256, 128 // factor, bilinear)
		self.up4 = Up(128, 64 // factor, bilinear)

		self.outc = OutConv(64, n_classes)

	def forward(self, x):
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)

		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		logits = self.outc(x)
		
		# softmax = torch.nn.Softmax(dim = 1)
		# op = softmax(logits)

		sm = torch.nn.Softmax(dim = 1)
		op = sm(logits)

		return op

if __name__ == "__main__":
	image = torch.rand((1, 3, 512, 256))
	model = Unet(3, 1)
	print(model(image).detach().numpy())

