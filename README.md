# Deep Vision

## Prerequisites

All required packages can be downloaded via pip:
```
pip -r install requirments.txt
```

The [Data](#data) chapter contains all relevant info on downloading the images with the corresponding annotations.



## Data

The [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) dataset was used for this project.
To get the images, run `scripts/download_data.py` and afterwards `data/unzip.sh`.
The annotations can be downloaded on the [InterHand2.6M website](https://mks0601.github.io/InterHand2.6M/).

The file structure look like this

```
$ data
|-- images
|   |-- train
|   |   |-- Capture0 ~ Capture26
|   |-- val
|   |   |-- Capture0
|   |-- test
|   |   |-- Capture0 ~ Capture7
|-- annotations
|   |-- skeleton.txt
|   |-- subject.txt
|   |-- train
|   |-- val
|   |-- test
```

## Training

The training process is documented in the `Training.ipynb` notebook. It contains all steps to train different architectures for keypoint detection.  


An alternative way to train the KeypointRCNN model and the FasterRCNN model would be to use the individual scripts.

```bash
#some scripts are needed from the torchvision repository, which we will download using a script
source ./setup_additional_scripts.sh

#start the training scripts 
python train_fasterrcnn_script.py
python train_keypointrcnn_script.py

#if u want to train in the background (like overnight on a remote server), use nohup
nohup python -u train_fasterrcnn_script &
nohup python -u train_keypointrcnn_script &
#afterwards u can close the terminal and check the status with `tail nohup.out`
```

## Visualization
The notebook `Visualize.ipynb` contains multiple functions to predict the keypoints using the models and afterwars visualize the results in the images.
