#Basic Imports
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import cv2
import matplotlib.pyplot as plt

#PyTorch
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
#Additional layers for KeypointRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNHeads
from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

#Own Code
from configs import training_configs
from scripts import create_yolo_annotations
from dataset import dataset,dataset_transforms

from engine import train_one_epoch, evaluate


train_data= dataset.Dataset(mode='train')
val_data  = dataset.Dataset(mode='val')
#test_data = dataset.Dataset(mode='test')

def collate_func(batch):
    return tuple(zip(*batch))

train_dataloader= DataLoader(train_data,batch_size = training_configs.BATCH_SIZE, shuffle = True, collate_fn=collate_func)
val_dataloader  = DataLoader(val_data,  batch_size = training_configs.BATCH_SIZE, shuffle = False, collate_fn=collate_func)
#test_dataloader = DataLoader(test_data, batch_size = training_configs.BATCH_SIZE, shuffle = False, collate_fn=collate_func)

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True, trainable_backbone_layers=2)
num_classes = 4 #right, left, interacting hands and background
# get number of input features (which will be passed from the backbone to the classifier)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one()
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)


num_out_channels = model.backbone.out_channels
keypoint_layers = tuple(512 for _ in range(8))
model.roi_heads.keypoint_head = KeypointRCNNHeads(num_out_channels, keypoint_layers)
num_keypoints = 42 #21 per hand
keypoint_dim_reduced = 512
model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(keypoint_dim_reduced,num_keypoints)


#move model to cuda if available 
if torch.cuda.is_available():
    print("GPU will be used")
    device = torch.device('cuda') 
else:
    print("No GPU available")
    device = torch.device('cpu')

model.to(device)


# optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# learning rate scheduler, decreaing learning rate 10x every epoch
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.1)


for epoch in range(training_configs.EPOCHS):
    # train for one epoch, printing every 10 iterations
    train_one_epoch(model, optimizer, train_dataloader, device, epoch, print_freq=100)
    # evaluate on the val dataset
    #evaluate(model, val_dataloader, device=device)
    # update the learning rate
    lr_scheduler.step()
    #save after each epoch
    torch.save(model.state_dict(), f'keypoint_rcnn_model_{epoch}.pt')
