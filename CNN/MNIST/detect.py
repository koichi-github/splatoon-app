import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from models import CNN
from PIL import Image

BATCH_SIZE = 100
WEIGHT_DECAY = 0.005
LEARNING_RATE = 0.0001
EPOCH = 11
PATH = "Dataset"


transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,), (0.5,))])

tensor = torchvision.transforms.ToTensor()

trainset = torchvision.datasets.MNIST(root = PATH, train = True, download = True, transform = transform)
# print(trainset[0])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = CNN()
net = net.to(device)
net.load_state_dict(torch.load('checkpoints/ckpt_e10.pth', map_location=device))
criterion = nn.CrossEntropyLoss()

input_ = Image.open("./img_1.jpg")
label = 0

# print(np.array(input_))

input_ = transform(input_)
# print(input_)
input_ = input_.unsqueeze(0)

input_ = input_.to(device)
outputs = net(input_)
print(input_)
print(outputs)
loss = criterion(outputs, tensor([label]))
_, predicted = outputs.max(1) 
        
    