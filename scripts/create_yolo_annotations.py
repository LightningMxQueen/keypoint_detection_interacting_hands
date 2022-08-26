import numpy as np 
import pandas as pd

import os 
import pathlib
import shutil

from configs import path_configs

hand_type_lookup = {
    "right":0,
    "left":1,
    "interacting":2
}

def copy_images_and_create_annotations(mode:str='train')-> None:
    """Combines the steps of copying the images and creating the annotation files 
    for the exitsing yolov5 implementation

    Args:
        mode (str, optional): 'train','test' or 'test'. Defaults to 'train'.
    """
    create_folders(mode=mode)
    copy_images(mode=mode)
    create_annotations(mode=mode)

def create_folders(mode:str = 'train')-> None:
    """Create the new folder(if not existing)

    Args:
        mode (str, optional): _description_. Defaults to 'train'.
    """
    base_path = f'{path_configs.YOLO_DATASET_PATH}/{mode}'
    #base path
    pathlib.Path(base_path).mkdir(parents=True,exist_ok=True)
    #label path
    pathlib.Path(f'{base_path}/images').mkdir(parents=True,exist_ok=True)
    #images path
    pathlib.Path(f'{base_path}/labels').mkdir(parents=True,exist_ok=True)

def copy_images(mode:str)-> None:
    """The Images of InterHand2.6 are in nested folders. 
    To avoid the problem, move the data into the newly created folder.
    This needes extra space ~80GB on hard drive, but reduces the code complexity in the existing yolov5 implementation

    Old folder structure is something like
    
    |-- images (base_path)
    |   |-- ${mode}
    |   |   |-- Capture0 
    |   |   |   |-- Situation
    |   |   |   |   |-- cam400000
    |   |   |   |   |   |-- image123.jpg
    |   |   |-- Capture2 
    ...
    |   |   |-- Capture3 
    ...

    Args:
        mode (str, optional): 'train','test' or 'test'. Defaults to 'train'.
    """
    base_path = f'{path_configs.IMG_PATH}/{mode}'
    base_path_len = len(base_path)
    output_path = f'{path_configs.YOLO_DATASET_PATH}/{mode}/images'
    #get all images inside the nested folders
    counter=0
    for path, subdirs, files in os.walk(base_path):
        for name in files:
            img_path = os.path.join(path, name)
            #path of image without the 'base_path' part and combination of the nested folder structure
            output_file_name = img_path[base_path_len+1:].replace('\\','_')
            #add some logging to check process status
            counter += 1
            if counter%10000 == 0:
                print(f'Copy file number {counter}')
            #copy
            shutil.copy( img_path, f'{output_path}/{output_file_name}')


def create_annotations(mode:str='train') -> None:
    """Creates annotation files for yolov5 in pytorch.
    See https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data for more info.
    Each image needs a own *.txt with the class label and bounding box.

    Args:
        mode (str, optional): 'train','test' or 'test'. Defaults to 'train'.
    """
    annot_path = f'{path_configs.ANNOTATION_PATH}/{mode}/InterHand2.6M_{mode}_data.json'
    output_base_path = f'{path_configs.YOLO_DATASET_PATH}/{mode}/labels'

    #read annotation file and split into 'annotation' and 'images'
    df_annot = pd.read_json(annot_path)
    #image_info part out of the annotation file 
    df_image_info = pd.json_normalize(df_annot['images'])[['id','file_name','width','height']]
    #annotation_info out of the annotation file 
    df_annot = pd.json_normalize(df_annot['annotations'])[['image_id','bbox','hand_type']]
    df_annot = df_annot.rename(columns={'image_id':'id'})

    #merge the dataframes -> single df with file_name, bbox and hand_type
    df = pd.merge(df_image_info, df_annot, on='id')
    #free RAM
    del df_annot
    del df_image_info

    #iterate through the annotations and create the required *.txt files
    #pandas is a column store, thus using zip(col1, col2 ...) is faster than e.g. df.iterrows()
    for file_name,bbox,width,height,hand_type in zip(df['file_name'],df['bbox'],df['width'],df['height'],df['hand_type']):
        #match copied image name
        output_file_name = file_name.replace('/','_').replace('.jpg','.txt')
        hand_class = hand_type_lookup[hand_type]
        line = f'{hand_class} {normalize_bbox(bbox=bbox,width=width,height=height)}'
        with open(f'{output_base_path}/{output_file_name}','w') as f:
            f.write(line)


def normalize_bbox(bbox:list,width:int,height:int) -> str:
    """create the bbox string used in the labels/*.txt files
    
    Args:
        bbox (list): [xmin, ymin, width, height]
        width (int): width of the image
        height (int): height of the image

    Returns:
        str: bbox coordinates as string
    """
    center_x = (bbox[0] + 0.5*bbox[2])/width
    center_y = (bbox[1] + 0.5*bbox[3])/height
    box_width = bbox[2]/width
    box_height = bbox[3]/height
    #transform to string
    return f'{center_x} {center_y} {box_width} {box_height}'

if __name__ == '__main__':
    copy_images_and_create_annotations(mode='train')
    copy_images_and_create_annotations(mode='val')
    copy_images_and_create_annotations(mode='test')

    