from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetMapper   # the default mapper
import os
import cv2
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_train_loader

def mapper(dataset_dict):
        img_path = os.path.join( os.getcwd(),"train","images",os.path.basename( dataset_dict["file_name"]))
        
        image = cv2.imread(img_path)
        
        annos =dataset_dict.pop("annotations")
        
        return {
        # create the format that the model expects
        "image": image,
        #"instances": utils.annotations_to_instances(annos, image.shape[1:])
        "instances": utils.annotations_to_instances(annos, image.shape[1:])
        }
        

class TrainerEnhance(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        '''
        if cfg.INPUT.ENHANCE_IMAGE:
            mapper = DatasetMapper(cfg, is_train=True, augmentations=[map_enhance])
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)
        '''
        return build_detection_train_loader(cfg, mapper=mapper)

if __name__ =="__main__":
    from utils import *
        
    config_file_path="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    checkpoint_url ="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

    output_dir = "./output"

    num_class = 6; #TODO:類別數量
    device = "cuda"

    train_dataset_name = "sb_train"
    train_images_path = "./train/images"
    train_json_annot_path = "./train/result.json"
    
    test_dataset_name="sb_test"
    cfg=get_train_cfg(config_file_path , checkpoint_url , train_dataset_name, test_dataset_name , num_class , device , output_dir)
    trainer = TrainerEnhance(cfg)