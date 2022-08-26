import torch 
import torch.nn as nn

from torchvision import models

def get_keypoint_model(num_keypoints:int = 42):
    """Return a model, that uses a vgg backbone for detecting keypoints

    Args:
        num_keypoints (int, optional): 21 for single hand, 42 for interacting hands. Defaults to 42.
    """
    #VGG16 as backbone. The layers need to be frozen  
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    #Change the Pooling of vgg model
    model.avgpool = nn.Sequential(
        nn.Conv2d(512,512,3),
        nn.MaxPool2d(2),
        nn.Flatten())

    #the classification head of the vgg model will be converted into 
    #into the keypoint regression head, with x,y for each keypoint
    #thus the num of outputs needs to be num_keypoint*2 (x,y for each keypoint) 
    model.classifier = nn.Sequential(
        nn.Linear(2048,512),
        nn.ReLU(), 
        nn.Dropout(0.1),
        nn.Linear(512,512),
        nn.ReLU(),
        nn.Linear(512, num_keypoints*2 ),
        nn.Sigmoid()
    )
    return model