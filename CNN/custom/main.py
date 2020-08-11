from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn

from trainer import fit
cuda = torch.cuda.is_available()

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from time import time

from networks import ResNet18
from datasets import MyDataset


cfg_train_path = "./data/sample/special/train.txt"
cfg_test_path = "./data/sample/special/test.txt"

data_dirname = cfg_train_path.split("/")[-2]

transform = transforms.ToTensor()
train_dataset = MyDataset(cfg_train_path, transform=transform, imgpath=True)
test_dataset = MyDataset(cfg_test_path, transform=transform, imgpath=True)


batch_size = 10
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the network and training parameters
num_out = 2
model = ResNet18(num_out)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
lr = 1e-3
optimizer = optim.Adam(model.parameters(), lr=lr)
n_epochs = 10
log_interval = 10
save_epoch_interval = 10


st = time()
fit(train_loader, test_loader, model, loss_fn, optimizer, n_epochs, device, log_interval, save_epoch_interval, data_dirname, num_out)
print(f"all time = {time() - st}")