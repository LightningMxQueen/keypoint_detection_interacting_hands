import numpy as np 
import torch
import torch.utils.data
import torchvision.transforms as transforms
from pycocotools.coco import COCO
import json
from PIL import Image

from configs import path_configs,default_configs

from dataset import preprocessing,coord_transforms

class Dataset(torch.utils.data.Dataset):

    def __init__(self,mode:str='train', transform=None, only_keypoints:bool = False, limit_handedness:str = 'all') -> None:
        #configs for dataset
        self.mode = mode #train,test,val
        self.transform = transforms.Compose([transforms.ToTensor()]) if transform is None else transform
        
        #paths
        self.img_path = path_configs.IMG_PATH
        self.annot_path = path_configs.ANNOTATION_PATH
        
        #info about joints
        self.num_joints = default_configs.NUM_JOINTS #21 for each hand
        self.root_joint_idx = {'right':20,'left':41} #id of the wrists
        self.joint_type = {'right': np.arange(0,self.num_joints), 'left': np.arange(self.num_joints,self.num_joints*2)}
        self.skeleton = preprocessing.load_skeleton(path_configs.SKELETON_PATH)

        #create datalists and fill them using the load_annotations()-method
        self.limit_handedness = limit_handedness
        self.datalist = []
        self.sequence_names = set() #use set to make each element unique in it
        self.load_datapoints()

        self.only_keypoints = only_keypoints

    def __len__(self) -> int:
        """each datapoint is stored in datalist,thus we can return it's length"""
        return len(self.datalist)

    def __getitem__(self,idx):
        if self.only_keypoints:
            #crop image and return normalized keypoints as target
            return self.keypoints__getitem__(idx)
        #return image + pycocotools like targets
        return self.total__getitem__(idx)

    def total__getitem__(self,idx):
        data = self.datalist[idx]

        #Read PIL Image and convert to tensor
        image = Image.open(data['img_path'])
        image = self.transform(image)

        #since we only have 1 bbox per per image, we can manually add another dimension
        #because this should return a list of labels/boxes/keypoints (e.g for multiple boxes per image)
        boxes = torch.as_tensor(np.expand_dims(data['bbox'],axis=0), dtype = torch.float32)
        labels = torch.as_tensor([data['hand_type_label']],dtype = torch.int64)
        keypoints = torch.as_tensor(np.expand_dims(data['keypoints'],axis=0),dtype = torch.float32)
        area = torch.as_tensor( [data['area']] , dtype = torch.float32)
        iscrowd = torch.as_tensor( [data['iscrowd']] , dtype = torch.int64)

        target = {
            'boxes':boxes,
            'labels':labels,
            'keypoints':keypoints,
            'area':area,
            'image_id':torch.as_tensor([idx],dtype = torch.int64),
            'iscrowd':iscrowd
        }

        return image,target

    def keypoints__getitem__(self,idx):
        data = self.datalist[idx]
        
        #Read PIL Image, Crop according to BBox and convert to tensor
        image = Image.open(data['img_path'])
        image = image.crop(data['bbox'])
        image = self.transform(image)

        # normalized keypoints
        keypoints = data['cropped_keypoints']
        
        #if only a hand type was selected, return relevant keypoints
        if self.limit_handedness in ('right', 'left', 'single'):
            if data['hand_type'] == 'right':
                keypoints = keypoints[:21]
            elif data['hand_type'] == 'left':
                keypoints = keypoints[21:]
        
        return image, torch.flatten(torch.tensor(keypoints,dtype=torch.float32))
    

    def load_datapoints(self) -> None:
        """iterate through the annot files and collect info about each datapoints.
        Store the results in datalists of class objects."""

        annot_file_base = f'{self.annot_path}/{self.mode}/InterHand2.6M_{self.mode}'
        
        #load all annot files
        db = COCO(f'{annot_file_base}_data.json') #contains img_info and bbox
        
        #camera and joint annot files contain info about the 3D Keypoints
        #Since we are interested in the 2D Keypoints, we need to convert them to 2D
        with open(f'{annot_file_base}_camera.json') as f:
            cameras = json.load(f)
        with open(f'{annot_file_base}_joint_3d.json') as f:
            joints = json.load(f)
        for annot_id in db.anns.keys():
            #get 'annotation' object out of .json-file
            annot = db.anns[annot_id]
            img_id = annot['image_id']
            #get 'image' object out of .json-file
            img_info = db.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_path = f"{self.img_path}/{self.mode}/{ file_name }"
            #metadata
            sequence_name = img_info['seq_name']
            self.sequence_names.add(sequence_name)

            #handedness label
            hand_type = annot['hand_type']
            hand_type_valid = np.array((annot['hand_type_valid']), dtype=np.float64)
            hand_type_label = preprocessing.convert_hand_type(hand_type)
            iscrowd = 1 if hand_type == 'interacting' else 0
            #bbox label
            img_height, img_width = img_info['height'],img_info['width'] #should be always the same (512x334), but we still double check it 
            if img_height != 512 or img_width != 334:
                continue           
            area = preprocessing.get_area(annot['bbox'])
            bbox = np.array(preprocessing.convert_bbox(annot['bbox']), dtype=np.float64)
            #bbox = np.array(annot['bbox'],dtype=np.float32) #x,y,width,height

            #info about capture -> needed for 3D to 2D conversion
            capture_id = img_info['capture']
            cam = img_info['camera']
            frame_idx = img_info['frame_idx']
            #camera matrix info -> intrinsic for every camera, were measured while capturing the images
            campos = np.array(cameras[str(capture_id)]['campos'][str(cam)], dtype=np.float64) # [x,y,z] coordinates of cam position
            camrot = np.array(cameras[str(capture_id)]['camrot'][str(cam)], dtype=np.float64) # 3x3 array, camera rotation matrix
            focal  = np.array(cameras[str(capture_id)]['focal'][str(cam)], dtype=np.float64)  # [focal_x, focal_y] of camera
            princpt= np.array(cameras[str(capture_id)]['princpt'][str(cam)], dtype=np.float64)# [principal_x, principal_y] of principal point
            #conversion  3D -> 2D
            joint_world = np.array(joints[str(capture_id)][str(frame_idx)]['world_coord'], dtype=np.float64) #Jx3 -> 3D joint coordinates in world coordinates (Unit: mm)
            joint_cam = coord_transforms.world2cam(joint_world.transpose(1,0), camrot, campos.reshape(3,1)).transpose(1,0) #transformation from world coords into cam coords
            joint_img = coord_transforms.cam2pixel(joint_cam, focal, princpt)[:,:2] #projection from cam coords(3D) onto 2D image

            #check if joints are valid
            joint_valid = np.array(annot['joint_valid'],dtype=np.float64).reshape(self.num_joints*2)
            # if root is not valid -> root-relative 3D pose is also not valid. Therefore, mark all joints as invalid
            joint_valid[self.joint_type['right']] *= joint_valid[self.root_joint_idx['right']]
            joint_valid[self.joint_type['left']] *= joint_valid[self.root_joint_idx['left']]
            keypoints = preprocessing.combine_keypoints(joint_img,joint_valid)
            #keypoints for the keypoint layer
            keypoints_non_cropped = keypoints.copy()
            cropped_keypoints = preprocessing.normalize_keypoints(keypoints= keypoints, bbox=bbox)
            
            data = {
                'img_path':img_path,
                'file_name':file_name,
                'seq_name':sequence_name,
                'hand_type':hand_type,
                'hand_type_valid':hand_type_valid,
                'hand_type_label':hand_type_label,
                'bbox':bbox,
                'keypoints':keypoints_non_cropped,
                'area':area,
                'iscrowd': iscrowd,
                'cropped_keypoints':cropped_keypoints,
            }

            #add datapoint list if it matches the limit 
            if self.limit_handedness == 'all':
                self.datalist.append(data)
            else:
                #label single and hand is either left or right
                if self.limit_handedness == 'single' and hand_type in ('right','left'):
                    self.datalist.append(data)
                #e.g. limit='interacting' and hand_type also, then append
                #e.g. limit='right' and hand_type='interacting' -> dont append
                elif hand_type == self.limit_handedness:
                    self.datalist.append(data)

    

    