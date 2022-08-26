ROOT_PATH= '.'

IMG_PATH = f'{ROOT_PATH}/data/images'
ANNOTATION_PATH = f'{ROOT_PATH}/data/annotations'

SKELETON_PATH = f'{ANNOTATION_PATH}/skeleton.txt'

YOLO_DATASET_PATH = f'{ROOT_PATH}/yolo_data'


MODEL_PATH = f'{ROOT_PATH}/models'

YOLO_MODEL_PATH = f'{MODEL_PATH}/yolov5s.pt'
KEYPOINT_RCNN_PATH = f'{MODEL_PATH}/keypoint_rcnn_model.pt'
FASTER_RCNN_PATH = f'{MODEL_PATH}/faster_rcnn_model.pt'
KEYPOINT_INTERACTING_PATH = f'{MODEL_PATH}/keypoint_interacting_model.pt'
KEYPOINT_SINGLE_PATH = f'{MODEL_PATH}/keypoint_single_model.pt'