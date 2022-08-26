import numpy as np
import PIL

from configs import path_configs,default_configs, model_configs

def load_skeleton(path:str=path_configs.SKELETON_PATH) -> list[dict]:
    """load skeleton file from the InterHand2.6M dataset (from the annotation folder)

    Args:
        path (str): path of sekelton.txt file 

    Returns:
        list[dict]: list of dict for each joint, with name, parent_id, child_id 
    """
    #create dict of every line (=joint) in skeleton file
    skeleton = []
    with open(path) as f:
        next(f) #skip first line with description
        for line in f:
            joint = {}
            joint_name, joint_id , joint_parent_id = line.split(' ')
            joint['name'] = joint_name
            joint['id'] = int(joint_id) 
            joint['parent_id'] = int(joint_parent_id)
            skeleton.append(joint)

    #connect child_ids
    #check for every joint, which other joints reference it
    for joint in skeleton:
        own_id = joint['id']
        child_ids = []
        for child_joint in skeleton:
            if child_joint['parent_id'] == own_id:
                child_ids.append(child_joint['id'])
        joint['child_id'] = child_ids
    return skeleton

def load_img(path:str) -> np.array:
    """Load image at given path using Pillow

    Args:
        path (str): path of the image

    Returns:
        np.array: image as np.array
    """
    img = PIL.Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32)
    return img

def lookup_handedness(hand_type:str)->np.array:
    """Convert handedness string to np array
    We can't use a dict for lookup, because then each datapoint would reference the same np array,
    thus we return a new one each time this method is called

    Args:
        hand_type (str): 'right','left' or 'interacting'

    Returns:
        np.array: np array where arr[0] encodes right hand, arr[1] encodes left hand. [1,1] means both hands -> interacting
    """
    if hand_type == 'right':
        return np.array([1,0], dtype=np.float32)
    elif hand_type == 'left':
        return np.array([0,1], dtype=np.float32)
    elif hand_type == 'interacting':
        return np.array([1,1], dtype=np.float32)

def convert_bbox(bbox:list[float]) -> list[float]:
    """convert bboxes from [x,y,width,height] into [x0,y0,x1,y1]

    Args:
        bbox (list[float]): box in format [x,y,width,height]

    Returns:
        list[float]: box in format [x0,y0,x1,y1]
    """
    area = bbox[2]*bbox[3]
    x1 = bbox[0] + bbox[2]
    y1 = bbox[1] + bbox[3]
    return [bbox[0],bbox[1],x1,y1]

def get_area(bbox:list[float]) -> float:
    """calculates  area of given bbox

    Args:
        bbox (list[float]): box in format [x,y,width,height]

    Returns:
        float: area of bbox
    """
    return bbox[2]*bbox[3]


def combine_keypoints(joints_image:np.array, joints_valid:np.array) -> np.array:
    """Combines coordinates with validity to create [x,y,valid] for each keypoint

    Args:
        joints_image (np.array): koordinates
        joints_valid (np.array): valid (in image)

    Returns:
        np.array: [x,y,valid] for each (42) keypoint
    """
    arr = np.column_stack((joints_image,joints_valid))
    #if keypoint is not valid, set the coordinates of the keypoint to [x=0,y=0]
    arr[:,0] *= arr[:,2]
    arr[:,1] *= arr[:,2]
    return arr

def convert_hand_type(hand_type:str) -> int:
    """Convert hand type to class"""
    if hand_type == 'right': 
        return 1
    elif hand_type == 'left': 
        return 2
    #hand type must be interacting
    else:
        return 3

def normalize_keypoints(keypoints:np.array,bbox:np.array) -> np.array:
    """Keypoints need to be normalized (range 0...1) after bbox cutout.

    Args:
        keypoints (np.array): Keypoints in format [x,y,visibility] for each keypoint
        bbox (np.array): BBox coord as [x1,y1,x2,y2]

    Returns:
        np.array: Normalized keypoints
    """

    #get keypoint distance to bbox 
    keypoints[:,0] -= bbox[0]
    keypoints[:,1] -= bbox[1]
    
    #normalize
    new_width = bbox[2] - bbox[0]
    new_height = bbox[3] - bbox[1]
    
    size = np.array([new_width,new_height],dtype=np.float64)
    keypoints[:,0:2] = keypoints[:,0:2]/size
    keypoints = keypoints[:,0:2]
    #set all keypoints to inbound if they aren't
    keypoints[keypoints<0]=0.0
    keypoints[keypoints>1]=1.0
    return keypoints