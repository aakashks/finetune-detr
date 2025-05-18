#!/usr/bin/env python
# coding: utf-8

# ## SwinTransformer RCNN training and evaluation code 
# This code is based on https://github.com/xiaohu2015/SwinT_detectron2<br>
# 
# I splitted video 0 and 1 to training and video 2 to validation.<br>
# So far, I could only train the model for 5 epochs and only achieved:<br>
# **bbox/AP = 15.2<br>**
# **bbox/AP50 = 35.7<br>**
# **bbox/AP75 = 8.0<br>**
# on the validation set.
# 
# However, I'd like to share my work and will continue working on it to further improve.

# ## Install requirements
# install Detectron2, timm and Detectron2 version implementation of SwinTransformer

import os
from pathlib import Path
from ast import literal_eval

import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image

from detectron2.data import (DatasetCatalog, 
                             MetadataCatalog, 
                             build_detection_test_loader
                            )
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, default_setup, hooks
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, inference_on_dataset

from swin.swint import add_swint_config

logger = setup_logger()


# ## 1. Load data

# ## 2.1 Define a dataset function

import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from PIL import Image # For getting image dimensions

# Your categories definition
categories_full = [
    {"id": 0, "name": "biker", "supercategory": "objects"},
    {"id": 1, "name": "car", "supercategory": "objects"},
    {"id": 2, "name": "pedestrian", "supercategory": "objects"},
    {"id": 3, "name": "trafficlight", "supercategory": "objects"},
    {"id": 4, "name": "trafficlight-Green", "supercategory": "objects"},
    {"id": 5, "name": "trafficlight-GreenLeft", "supercategory": "objects"},
    {"id": 6, "name": "trafficlight-Red", "supercategory": "objects"},
    {"id": 7, "name": "trafficlight-RedLeft", "supercategory": "objects"},
    {"id": 8, "name": "trafficlight-Yellow", "supercategory": "objects"},
    {"id": 9, "name": "trafficlight-YellowLeft", "supercategory": "objects"},
    {"id": 10, "name": "truck", "supercategory": "objects"},
    {"id": 11, "name": "Arret", "supercategory": "objects"}
]
# Extract just the names for Detectron2 metadata
THING_CLASSES = [cat['name'] for cat in categories_full]

# Create a mapping from COCO category_id to a 0-indexed Detectron2 category_id if necessary
# If your COCO category_ids are already 0-indexed and contiguous, this might be simpler.
# For this example, we'll assume your annotations.json uses the 'id' from categories_full.
COCO_CATEGORY_ID_TO_DETECTRON2_ID = {cat['id']: i for i, cat in enumerate(categories_full)}
# If your annotations.json already uses 0, 1, 2... for category_id that match the order in THING_CLASSES,
# then COCO_CATEGORY_ID_TO_DETECTRON2_ID = {i: i for i in range(len(THING_CLASSES))}


def get_custom_coco_dicts(img_dir, annotation_file_path):
    """
    Args:
        img_dir (str): path to the image directory.
        annotation_file_path (str): path to the COCO format annotation json file.

    Returns:
        list[dict]: a list of dicts in Detectron2 dataset format.
    """
    with open(annotation_file_path, 'r') as f:
        coco_data = json.load(f)

    dataset_dicts = []
    
    # Create a mapping from image_id to image info
    images_info = {img['id']: img for img in coco_data['images']}
    
    # Create a mapping from image_id to annotations
    annotations_by_image_id = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image_id:
            annotations_by_image_id[img_id] = []
        annotations_by_image_id[img_id].append(ann)

    for image_id, img_info in images_info.items():
        record = {}
        
        # Prefer file_name from COCO json if it includes path, otherwise construct it
        if 'file_name' in img_info:
            # Check if file_name is just the name or includes subdirs
            # Assuming your img_dir is the root and file_name might be like "subdir/image.jpg"
            # or just "image.jpg"
            filename = os.path.join(img_dir, img_info['file_name'])
            # If your 'img_dir' in torchvision.datasets.CocoDetection was the parent of 'images'
            # and annotations.json was in 'img_dir', then img_dir here should point to the 'images' subfolder.
            # Example: if original img_folder was 'my_dataset/' (containing 'images/' and 'annotations.json')
            # then img_dir for this function should be 'my_dataset/images/'
        else:
            # This case should ideally not happen if coco_data['images'] is complete
            print(f"Warning: 'file_name' not found for image_id {image_id}. Skipping.")
            continue

        # If image dimensions are not in coco_data['images'], load the image to get them
        # It's better if they are pre-computed in the JSON.
        height = img_info.get('height')
        width = img_info.get('width')
        if height is None or width is None:
            try:
                with Image.open(filename) as pil_img:
                    width, height = pil_img.size
            except FileNotFoundError:
                print(f"Warning: Image file not found at {filename} for image_id {image_id}. Skipping.")
                continue


        record["file_name"] = filename
        record["image_id"] = image_id # Use the original COCO image_id
        record["height"] = height
        record["width"] = width
      
        objs = []
        if image_id in annotations_by_image_id:
            for ann in annotations_by_image_id[image_id]:
                # COCO bbox is [x, y, width, height]
                # Detectron2 expects BoxMode.XYWH_ABS for this format
                coco_bbox = ann['bbox']
                
                # Get the original category_id from your COCO annotation
                original_category_id = ann['category_id']
                
                # Map it to your 0-indexed Detectron2 category_id
                # If your category_ids in the JSON are NOT the same as in `categories_full` `id` field,
                # you will need to adjust this mapping.
                # This assumes ann['category_id'] corresponds to the 'id' field in categories_full.
                try:
                    d2_category_id = COCO_CATEGORY_ID_TO_DETECTRON2_ID[original_category_id]
                except KeyError:
                    print(f"Warning: Unknown category_id {original_category_id} in annotations for image {image_id}. Skipping annotation.")
                    continue

                obj = {
                    "bbox": coco_bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": d2_category_id,
                    "iscrowd": ann.get("iscrowd", 0) # Default to 0 if not present
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


YOUR_TRAIN_IMG_FOLDER = "your_actual_train_image_folder_path" # e.g., /kaggle/input/your_dataset/train/images
YOUR_TRAIN_ANN_FILE = "your_actual_train_annotations_json_path" # e.g., /kaggle/input/your_dataset/train/annotations.json
YOUR_VAL_IMG_FOLDER = "your_actual_val_image_folder_path" # e.g., /kaggle/input/your_dataset/val/images
YOUR_VAL_ANN_FILE = "your_actual_val_annotations_json_path" # e.g., /kaggle/input/your_dataset/val/annotations.json


# Registering the datasets
for d_type, img_folder, ann_file in [("train", YOUR_TRAIN_IMG_FOLDER, YOUR_TRAIN_ANN_FILE),
                                     ("val", YOUR_VAL_IMG_FOLDER, YOUR_VAL_ANN_FILE)]:
    dataset_name = f"my_custom_coco_{d_type}"
    # Ensure dataset is not already registered, or pop it first if you're re-running
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.pop(dataset_name)
    
    # The lambda function captures the current values of img_folder and ann_file
    DatasetCatalog.register(dataset_name, lambda f=img_folder, a=ann_file: get_custom_coco_dicts(f, a))
    MetadataCatalog.get(dataset_name).set(thing_classes=THING_CLASSES, # Your list of category names
                                           evaluator_type="coco", # For COCOEvaluator
                                           json_file=ann_file, # Path to the original COCO JSON (for COCOEvaluator)
                                           image_root=img_folder # Path to image folder (for COCOEvaluator)
                                          )
    print(f"Registered dataset: {dataset_name}")
    print(f"Associated metadata: {MetadataCatalog.get(dataset_name)}")


# ## 2.2 Check the Dataset
# visualize the dataset for verification

my_ds = DatasetCatalog.get("my_custom_train")
metadata = MetadataCatalog.get('my_custom_train')


for data in gbl_ds:
    if len(data['annotations']):
        break
im = cv2.cvtColor(cv2.imread(data['file_name']), cv2.COLOR_BGR2RGB)
v = Visualizer(im, 
               metadata=MetadataCatalog.get('my_custom_train'),
               scale=0.5)
out = v.draw_dataset_dict(data)
im = Image.fromarray(out.get_image())
im


# ## 3.1 Define a custom Trainer to evaluate on custom dataset

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name=dataset_name,
                             tasks=["bbox"],
                             distributed=True,
                             output_dir=output_folder)
    
    @classmethod
    def build_tta_model(cls, cfg, model):
        return GeneralizedRCNNWithTTA(cfg, model)
    
    @classmethod
    def test_with_TTA(cls, cfg, model):
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = self.build_tta_model(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


# ## 3.2 Set a config

TRAIN_STEPS = 4242 # only 4242 images with annotation
cfg = get_cfg()
add_swint_config(cfg)
cfg.merge_from_file('swin/configs/SwinT/faster_rcnn_swint_T_FPN_3x_.yaml')
cfg.DATASETS.TRAIN = ("my_custom_train",)
cfg.DATASETS.TEST = ("my_val_dataset",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = None
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.MAX_ITER = TRAIN_STEPS * 10
cfg.SOLVER.STEPS = []
cfg.SOLVER.CHECKPOINT_PERIOD = TRAIN_STEPS
cfg.TEST.EVAL_PERIOD = TRAIN_STEPS 
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)


# ## 3.3 Train
# 
# I could only run for 5 epochs due to runtime quota

# load pretrained weights
other_weights = torch.load('faster_rcnn_swint_T.pth')['model']
self_weight = trainer.model.state_dict()
for name, param in self_weight.items():
    if name in other_weights:
        if other_weights[name].shape == param.shape:
            self_weight[name] = other_weights[name]
        else:
            print(f"size mismatch at {name}")
    else:
        print(f"layer {name} does not exist")
trainer.model.load_state_dict(self_weight)
trainer.train()


# ## 4.1 Evaluate
# evaluate on validation set

trainer.model.load_state_dict(torch.load('/kaggle/input/swintransformer-with-detectron2/model_0021209.pth')['model'])
trainer.test(cfg, trainer.model)


# ## 4.2 Visualize a few prediction examples

val_ds = DatasetCatalog.get("my_val_dataset")
trainer.model.eval()
metadata = MetadataCatalog.get('my_custom_train')


for _ in range(5):
    idx = np.random.randint(0, len(val_ds))
    data = val_ds[idx]
    print(data['file_name'])
    im = cv2.imread(data['file_name'])
    im_tensor = torch.from_numpy(im).permute(2,0,1)  # h, w, c -> c, h, w
    h, w, _ = im.shape
    with torch.no_grad():
        pred = trainer.model([{"image": im_tensor.cuda(), "width": w, "height": h}])
    v = Visualizer(im[:, :, ::-1],
                   metadata=metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(pred[0]["instances"].to("cpu"))
    plt.figure()
    plt.imshow(out.get_image())




