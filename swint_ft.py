import os
import sys
from pathlib import Path
from ast import literal_eval
from collections import OrderedDict
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import torch
import torchvision
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


# ## SwinTransformer RCNN training and evaluation code 
#
# Fine-tuning SwinTransformer for object detection on a custom dataset.

# ## Install requirements
# install Detectron2, timm and Detectron2 version implementation of SwinTransformer

# Set up configuration parameters
seed = 42
num_workers = 8
dataset_path = "./workspace/iva/data/combined"
val_dataset_path = "./workspace/iva/data/55_all"
batch_size = 4  # Smaller batch size for Swin-T compared to DETR
val_batch_size = 8
lr = 1e-4
weight_decay = 1e-4
lr_backbone = 1e-5
max_epochs = 100
max_iterations = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%m-%d_%H-%M")

logger = setup_logger()


# ## 1. Load data using CocoDetection similar to DETR approach

# Category definitions for our dataset
id2label = {
    0: "biker",
    1: "car",
    2: "pedestrian",
    3: "trafficlight",
    4: "trafficlight-Green",
    5: "trafficlight-Greenleft",
    6: "trafficlight-Red",
    7: "trafficlight-Redleft",
    8: "trafficlight-Yellow",
    9: "trafficlight-Yellowleft",
    10: "truck",
    11: "Arret"
}
label2id = {v: k for k, v in id2label.items()}

# Define CocoDetection dataset class for a more compatible approach with DETR
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, processor, train=True):
        ann_file = os.path.join(img_folder, "annotations.json")
        super().__init__(img_folder, ann_file)
        self.processor = processor

    def __getitem__(self, idx):
        # read in PIL image and target in COCO format
        img, target = super().__getitem__(idx)

        # preprocess image and target 
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        # For Detectron2 format
        record = {}
        record["file_name"] = self.root + "/" + self.coco.loadImgs(image_id)[0]["file_name"]
        record["image_id"] = image_id
        record["height"] = img.height
        record["width"] = img.width
        
        objs = []
        for anno in target['annotations']:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"],
                "iscrowd": anno.get("iscrowd", 0)
            }
            objs.append(obj)
        record["annotations"] = objs
        
        return record

# Import required additional libraries
import json

# Create a mapping from category_id to index
COCO_CATEGORY_ID_TO_DETECTRON2_ID = {k: i for i, k in id2label.items()}

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
        
        if 'file_name' in img_info:
            filename = os.path.join(img_dir, img_info['file_name'])
        else:
            print(f"Warning: 'file_name' not found for image_id {image_id}. Skipping.")
            continue

        # Get image dimensions
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
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
      
        objs = []
        if image_id in annotations_by_image_id:
            for ann in annotations_by_image_id[image_id]:
                # COCO bbox is [x, y, width, height]
                coco_bbox = ann['bbox']
                
                # Get category ID and map to Detectron2 format
                original_category_id = ann['category_id']
                
                try:
                    d2_category_id = COCO_CATEGORY_ID_TO_DETECTRON2_ID[original_category_id]
                except KeyError:
                    print(f"Warning: Unknown category_id {original_category_id} in annotations for image {image_id}. Skipping annotation.")
                    continue

                obj = {
                    "bbox": coco_bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": d2_category_id,
                    "iscrowd": ann.get("iscrowd", 0)
                }
                objs.append(obj)
        
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


# Define paths for training and validation data
TRAIN_IMG_FOLDER = dataset_path
TRAIN_ANN_FILE = os.path.join(dataset_path, "annotations.json")
VAL_IMG_FOLDER = val_dataset_path
VAL_ANN_FILE = os.path.join(val_dataset_path, "annotations.json")

# Register datasets for training and validation
for d_type, img_folder, ann_file in [("train", TRAIN_IMG_FOLDER, TRAIN_ANN_FILE),
                                     ("val", VAL_IMG_FOLDER, VAL_ANN_FILE)]:
    dataset_name = f"my_custom_{d_type}"
    # Remove if already registered
    if dataset_name in DatasetCatalog.list():
        DatasetCatalog.pop(dataset_name)
    
    # Register with appropriate function
    DatasetCatalog.register(dataset_name, lambda f=img_folder, a=ann_file: get_custom_coco_dicts(f, a))
    MetadataCatalog.get(dataset_name).set(
        thing_classes=list(id2label.values()),
        evaluator_type="coco",
        json_file=ann_file,
        image_root=img_folder
    )
    print(f"Registered dataset: {dataset_name}")

# Create dataset instances for direct use (similar to DETR approach)
train_dataset = CocoDetection(img_folder=TRAIN_IMG_FOLDER, processor=None) 
val_dataset = CocoDetection(img_folder=VAL_IMG_FOLDER, processor=None, train=False)

print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))

# Check the dataset - visualize a sample
my_ds = DatasetCatalog.get("my_custom_train")
metadata = MetadataCatalog.get("my_custom_train")

# Sample visualization
if len(my_ds) > 0:
    sample_idx = 0
    while sample_idx < len(my_ds):
        data = my_ds[sample_idx]
        if len(data['annotations']) > 0:
            break
        sample_idx += 1
    
    if sample_idx < len(my_ds):
        im = cv2.cvtColor(cv2.imread(data['file_name']), cv2.COLOR_BGR2RGB)
        v = Visualizer(im, metadata=metadata, scale=0.5)
        out = v.draw_dataset_dict(data)
        plt.figure(figsize=(10, 10))
        plt.imshow(out.get_image())


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
        model = cls.build_tta_model(cfg, model)
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

TRAIN_STEPS = 1000
cfg = get_cfg()
add_swint_config(cfg)
cfg.merge_from_file('swin/configs/SwinT/faster_rcnn_swint_T_FPN_3x_.yaml')
cfg.DATASETS.TRAIN = ("my_custom_train",)
cfg.DATASETS.TEST = ("my_custom_val",)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(id2label)  # Set to number of categories
cfg.MODEL.WEIGHTS = None
cfg.DATALOADER.NUM_WORKERS = num_workers
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = lr  # Learning rate from parameters
cfg.SOLVER.WEIGHT_DECAY = weight_decay  # Weight decay from parameters
cfg.SOLVER.MAX_ITER = max_iterations
cfg.SOLVER.STEPS = []  # No step schedule
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

# Model checkpoint path for evaluation
model_save_path = f"ckpts/swint-ft-{max_epochs}-epochs-{timestamp}"
checkpoint_path = os.path.join(model_save_path, "model_final.pth")

# If trained locally and ready to evaluate
if os.path.exists(checkpoint_path):
    trainer.model.load_state_dict(torch.load(checkpoint_path)['model'])
    results = trainer.test(cfg, trainer.model)
    print("Evaluation results:", results)
else:
    print(f"Checkpoint not found at {checkpoint_path}. Skipping evaluation.")


# ## 4.2 Visualize a few prediction examples

val_ds = DatasetCatalog.get("my_custom_val")
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

