from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode

import random
import cv2
import matplotlib.pylab as plt
import os

def plot_samples(dataset_name , n=1 ):
    dataset_custom = DatasetCatalog.get(dataset_name)
    dataset_custom_metadata = MetadataCatalog.get(dataset_name)
    
    for s in random.sample(dataset_custom,n):
        img_path = os.path.join( os.getcwd(),"train","images",os.path.basename( s['file_name']))
        img = cv2.imread(img_path) #cv load in BGR format
        print(img_path)
        v = Visualizer(img[:,:,::-1] , metadata=dataset_custom_metadata , scale=0.5) #deteron2 need RGB format. use ::-1 to swap red and blue channel
        v = v.draw_dataset_dict(s)
        plt.figure(figsize=(15,20))
        plt.imshow(v.get_image())
        plt.show()

def get_real_img_path(rootdir,file_name):
    img_path = os.path.join( os.getcwd(),rootdir,os.path.basename( file_name))
    return img_path
    
def get_train_cfg(config_file_path , checkpoint_url , train_dataset_name , test_dataset_name, num_classes , device , output_dir):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(config_file_path))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint_url)
    cfg.DATASETS.TRAIN = (train_dataset_name, )
    cfg.DATASETS.TEST = (test_dataset_name , )
    
    cfg.DATALOADER.NUM_WORKERS =2
    cfg.SOLVER.IMS_PER_BATCH =2 # 2 images per batch
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000 
    cfg.SOLVER.STEPS = []
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.DEVICE = device
    cfg.OUTPUT_DIR = output_dir
    
    return cfg
    
    