import json
import os
from typing import Optional

# Third-party imports
import pandas as pd
import pytorch_lightning as pl
import submitit
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions

# Standard library imports
from tqdm import tqdm
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

# Local application/library specific imports
from adapter_utils import *
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if "LD_LIBRARY_PATH" in os.environ:
    del os.environ["LD_LIBRARY_PATH"]


class CocoCaptionsDataset(CocoCaptions):

    def __init__(self, root, annFile, image_processor, image_ids):
        super().__init__(root, annFile)
        self.image_processor = image_processor
        self.ids = list(sorted(image_ids))

    def __getitem__(self, index):
        images, captions = super().__getitem__(index)
        pixel_values = self.image_processor(images=images,
                                            return_tensors="pt").pixel_values
        return {
            "raw_images": images,
            "pixel_values": pixel_values,
            "captions": captions,
            "image_id": self.ids[index],
        }


class CocoDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_dir,
        ann_dir,
        split_file,
        vision_model_name_or_path,
        collate_fn,
        train_batch_size=32,
        val_batch_size=512,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.ann_dir = ann_dir
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.collate_fn = collate_fn
        self.image_processor = ViTImageProcessor.from_pretrained(
            vision_model_name_or_path)
        self.split_file = split_file
        with open(split_file, "r") as f:
            self.split = pd.DataFrame(json.load(f)["images"])

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            train_ids = self.split[self.split["split"] == "train"].cocoid
            eval_ids = self.split[self.split["split"] == "val"].cocoid

            self.coco_train = CocoCaptionsDataset(
                self.data_dir + "/train2014",
                self.ann_dir + "captions_train2014.json",
                self.image_processor,
                train_ids,
            )

            self.coco_val = CocoCaptionsDataset(
                self.data_dir + "/val2014",
                self.ann_dir + "captions_val2014.json",
                self.image_processor,
                eval_ids,
            )

        if stage == "test" or stage is None:
            test_ids = self.split[self.split["split"] == "test"].cocoid
            self.coco_test = CocoCaptionsDataset(
                self.data_dir + "/val2014",
                self.ann_dir + "captions_val2014.json",
                self.image_processor,
                test_ids,
            )

    def train_dataloader(self):
        return DataLoader(
            self.coco_train,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            self.coco_val,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )

    def test_dataloader(self):
        return DataLoader(
            self.coco_test,
            batch_size=self.val_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        )


class ImageCaptioningModel(pl.LightningModule):

    def __init__(
        self,
        vision_model_name_or_path,
        text_model_name_or_path,
        data_module,
        peft_conf,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_model_name_or_path, text_model_name_or_path)

        freeze(self.model.encoder)
        self.peft_conf = peft_conf
        setup_ft(model=self.model, peft_conf=peft_conf)

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.data_module = data_module
        self.data_module.setup()

        self.learning_rate = learning_rate
        self.val_loss = AvgMetric()
        self.test_loss = AvgMetric()

    def forward(self, images, labels):
        return self.model(pixel_values=images, labels=labels)

    def pad_text(self, captions):
        encoded_inputs_list = [
            self.tokenizer(text,
                           padding=True,
                           truncation=True,
                           return_tensors="pt").input_ids.flatten().long()
            for text in captions
        ]
        return torch.nn.utils.rnn.pad_sequence(encoded_inputs_list,
                                               batch_first=True)

    def training_step(self, batch, batch_idx):
        images, captions = batch["pixel_values"], batch["captions"]
        labels = self.pad_text(captions).to(self.device)
        output = self(images=images.to(self.device), labels=labels)

        self.log(
            "loss",
            output.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            batch_size=len(captions),
        )
        return {"loss": output.loss}

    def validation_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        captions = batch["captions"]
        labels = self.pad_text(captions).to(self.device)
        output = self.model(pixel_values=images, labels=labels)
        if batch_idx == 0:
            results = self.evaluate(self.device)
            self.log_dict(results)

        self.val_loss.update(output.loss.item(), 1)

    def test_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        captions = batch["captions"]
        labels = self.pad_text(captions).to(self.device)
        output = self.model(pixel_values=images, labels=labels)
        if batch_idx == 0:
            results = self.evaluate(self.device)
            self.log_dict(results)
        self.test_loss.update(output.loss.item(), 1)

    def on_validation_epoch_end(self):
        self.log("val_loss", self.val_loss.compute())
        self.val_loss.reset()

    def on_test_epoch_end(self):
        self.log("test_loss", self.test_loss.compute())
        self.test_loss.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

    def evaluate(self, device):
        all_results = []
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            for batch in tqdm(self.data_module.test_dataloader()):
                images, image_ids = batch["pixel_values"], batch["image_id"]
                images = images.to(device)

                generated_ids = self.model.generate(images)
                generated_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                results = [{
                    "image_id": image_id,
                    "caption": caption
                } for image_id, caption in zip(image_ids, generated_text)]
                all_results.extend(results)
        return evaluate_coco(all_results)


def sweep_iteration(config=None):
    print("Starting sweep iteration")
    seed_everything(1)
    torch.set_float32_matmul_precision("high")

    with wandb.init():
        config = wandb.config if config is None else config
        print(config)

        peft_confg = create_config(
            args.ft_type,
            args.hidden_dim,
            prefix_length=10,
            scaling_factor=config.scale_factor
            if args.ft_type == "rnn" else 1.0,
            bidirectional=args.bidirectional,
            r=args.bottle_neck_ratio,
            alpha=args.alpha,
        )
        name = peft_confg.ft_type

        wandb_logger = WandbLogger()
        data_module = CocoDataModule(
            data_dir="/data/home/ngjhn/read/coco",
            ann_dir="/data/home/ngjhn/read/coco/annotations/",
            vision_model_name_or_path="google/vit-base-patch16-224-in21k",
            split_file="/data/home/ngjhn/read/coco/dataset_coco.json",
            collate_fn=collate_fn,
            train_batch_size=8,
            val_batch_size=128,
        )

        model = ImageCaptioningModel(
            vision_model_name_or_path="google/vit-base-patch16-224-in21k",
            text_model_name_or_path="bert-large-uncased",
            data_module=data_module,
            learning_rate=config.learning_rate,
            peft_conf=peft_confg,
        )

        trainer = pl.Trainer(
            max_epochs=40,
            devices=1,
            callbacks=[
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename=
                    "{name}-lr:{config.learning_rate:.4f}-{epoch}-{val_loss:.2f}",
                    dirpath=f"/data/home/ngjhn/read/lightning_logs/{name}",
                    auto_insert_metric_name=True,
                ),
                EarlyStopping(monitor="val_loss",
                              min_delta=0.00,
                              patience=3,
                              verbose=True),
                TQDMProgressBar(),
            ],
            default_root_dir="/data/home/ngjhn/read",
            logger=wandb_logger,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=2,
        )

        trainer.fit(model, data_module)
        return model


def sweep(sweep_config):
    sweep_id = (wandb.sweep(sweep_config, project="read")
                if not args.sweep_id else args.sweep_id)
    wandb.agent(sweep_id, project="read", function=sweep_iteration, count=10)


if __name__ == "__main__":
    # Submit a job to the SLURM queue, batch mode only. Find logs
    # in the conf.slurm_folder.
    args = parse_args()
    gpus_per_node = 1
    nodes = 1
    executor = submitit.AutoExecutor(folder=args.slurm_folder)
    executor.update_parameters(
        name=f"{args.name}",
        slurm_partition="lowpri",
        nodes=nodes,
        tasks_per_node=1,
        slurm_gpus_per_task=1,
        cpus_per_task=10 * gpus_per_node,
        slurm_time=6000,
    )
    if args.sweep:
        sweep_config = {
            "method": "bayes",
            "name": args.name,
            "metric": {
                "name": "val_loss",
                "goal": "maximize"
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 3
            },
            "parameters": {
                # "learning_rate": {
                #     "min": 1e-4,
                #     "max": 1e-3
                # }
                "learning_rate": {
                    "values": [
                        0.0000854010639,
                        0.0008540106397,
                        0.0009927153629,
                        0.0004784757354,
                        0.0006663023945,
                        0.0005272516063,
                    ]
                }
            },
        }
        if args.ft_type == "rnn":
            sweep_config["parameters"]["scale_factor"] = {
                "min": 0.01,
                "max": 1.0
            }

        with executor.batch():
            job = executor.submit(sweep, sweep_config)
        print(f"Submitted job {job.job_id} to SLURM queue")
    else:
        peft_confg = create_config(args.ft_type,
                                   args.hidden_dim,
                                   scaling_factor=args.scale_factor)
        with executor.batch():
            job = executor.submit(
                sweep_iteration,
                Config(learning_rate=args.learning_rate, peft_conf=peft_confg),
            )
        print(f"Submitted job {job.job_id} to SLURM queue")
