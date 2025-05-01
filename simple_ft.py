import logging
import os
import sys
from dataclasses import dataclass, field, make_dataclass
from datetime import datetime

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from PIL import Image
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset
from transformers import DetrForObjectDetection, DetrImageProcessor

seed = 5012025
num_workers = 8
train_val_split = 0.15
dataset_path = "/home/singh/workspace/iva/data/combined"
batch_size = 16
val_batch_size = 32
lr = 1e-4
weight_decay = 1e-4
lr_backbone = 1e-5
max_epochs = 100


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = load_dataset("json", data_files=f"{dataset_path}/metadata.jsonl")


train_val_split = None if "validation" in dataset.keys() else train_val_split
if isinstance(train_val_split, float) and train_val_split > 0.0:
    split = dataset["train"].train_test_split(train_val_split, seed=seed)
    dataset["train"] = split["train"]
    dataset["validation"] = split["test"]

# Get dataset categories and prepare mappings for label_name <-> label_id
# categories = dataset["train"].features["objects"].feature["category"].names
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


image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

class HFDetectionDataset(Dataset):
    def __init__(self, dataset, img_folder, processor, split="train"):
        self.dataset = dataset[split]  # HF dataset split
        self.img_folder = img_folder   # Root directory with images
        self.processor = processor
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get sample from HF dataset
        sample = self.dataset[idx]
        
        # Load image from file
        image_path = os.path.join(self.img_folder, sample["file_name"])
        img = Image.open(image_path).convert("RGB")
        
        # Format bboxes and annotations for DETR
        objects = sample["objects"]
        bboxes = objects["bbox"]  # List of bboxes [x, y, width, height]
        categories = objects["categories"]  # List of category IDs
        
        # Convert to COCO format
        annotations = []
        for i, (bbox, category_id) in enumerate(zip(bboxes, categories)):
            x, y, width, height = bbox
            
            annotation = {
                "id": objects["id"][i] if "id" in objects else i,
                "image_id": sample["image_id"],
                "bbox": bbox,  # [x, y, width, height]
                "category_id": category_id,
                "area": objects["areas"][i] if "areas" in objects else width * height,
                "iscrowd": 0
            }
            annotations.append(annotation)
        
        # Format target in DETR expected format
        target = {'image_id': sample["image_id"], 'annotations': annotations}
        
        # Process image and annotations
        encoding = self.processor(images=img, annotations=target, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
        target = encoding["labels"][0]  # remove batch dimension
        
        return pixel_values, target



train_dataset = HFDetectionDataset(dataset, dataset_path, image_processor, split="train")
val_dataset = HFDetectionDataset(dataset, dataset_path, image_processor, split="validation")


print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(val_dataset))


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = labels
    return batch


train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=val_batch_size, num_workers=num_workers)

class Detr(pl.LightningModule):
     def __init__(self, lr, lr_backbone, weight_decay):
         super().__init__()
         # replace COCO classification head with custom head
         # we specify the "no_timm" variant here to not rely on the timm library
         # for the convolutional backbone
         self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50",
                                                             revision="no_timm",
                                                             num_labels=len(id2label),
                                                             ignore_mismatched_sizes=True)
         # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
         self.lr = lr
         self.lr_backbone = lr_backbone
         self.weight_decay = weight_decay
         # self.config = make_dataclass("config", ['id2label', 'label2id'])(id2label=id2label, label2id=label2id)

     def forward(self, pixel_values, pixel_mask):
       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

       return outputs

     def common_step(self, batch, batch_idx):
       pixel_values = batch["pixel_values"]
       pixel_mask = batch["pixel_mask"]
       labels = [{k: v.to(self.device) for k, v in t.items()} for t in batch["labels"]]

       outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

       loss = outputs.loss
       loss_dict = outputs.loss_dict

       return loss, loss_dict

     def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k,v in loss_dict.items():
          self.log("train_" + k, v.item())

        return loss

     def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k,v in loss_dict.items():
          self.log("validation_" + k, v.item())

        return loss

     def configure_optimizers(self):
        param_dicts = [
              {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
              {
                  "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
                  "lr": self.lr_backbone,
              },
        ]
        optimizer = torch.optim.AdamW(param_dicts, lr=self.lr,
                                  weight_decay=self.weight_decay)

        return optimizer

     def train_dataloader(self):
        return train_dataloader

     def val_dataloader(self):
        return val_dataloader


model = Detr(lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay)


trainer = Trainer(max_epochs=max_epochs, gradient_clip_val=0.1)
trainer.fit(model)


from transformers import Trainer as TransformersTrainer
from transformers import TrainingArguments

timestamp = datetime.now().strftime("%m-%d_%H-%M")
model_save_path = f"ckpts/detr-simple-2-ft-{max_epochs}-epochs-{timestamp}"
trainer = TransformersTrainer(model=model.model, processing_class=image_processor, data_collator=collate_fn, args=TrainingArguments(output_dir=model_save_path))
trainer.save_model()


# model = Detr.load_from_checkpoint('./lightning_logs/version_5/checkpoints/epoch=368-step=38745.ckpt', lr=lr, lr_backbone=lr_backbone, weight_decay=weight_decay).to(device)
# model.eval();


