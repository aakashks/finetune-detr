from dataclasses import make_dataclass
from datetime import datetime

import pytorch_lightning as pl
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor

image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


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
    11: "Arret",
}


class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        # replace COCO classification head with custom head
        # we specify the "no_timm" variant here to not rely on the timm library
        # for the convolutional backbone
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            revision="no_timm",
            num_labels=len(id2label),
            ignore_mismatched_sizes=True,
        )
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

        outputs = self.model(
            pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels
        )

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("train_" + k, v.item())

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("validation_" + k, v.item())

        return loss

    def configure_optimizers(self):
        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": self.lr_backbone,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )

        return optimizer


model = Detr.load_from_checkpoint(
    "./lightning_logs/version_12/checkpoints/epoch=9-step=2350.ckpt",
    lr=1e-4,
    lr_backbone=1e-5,
    weight_decay=1e-4,
)

epochs = 9
from transformers import Trainer, TrainingArguments

timestamp = datetime.now().strftime("%m-%d_%H-%M")
model_save_path = f"ckpts/detr-simple-2-ft-{epochs}-epochs-{timestamp}"
trainer = Trainer(
    model=model.model,
    processing_class=image_processor,
    data_collator=collate_fn,
    args=TrainingArguments(output_dir=model_save_path),
)
trainer.save_model()
