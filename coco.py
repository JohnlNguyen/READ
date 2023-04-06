# Standard library imports
from tqdm import tqdm
from torchmetrics.metric import Metric
import argparse
import json
import os
from collections import defaultdict
from typing import NamedTuple, Optional, Any

# Third-party imports
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (DeviceStatsMonitor, ModelCheckpoint,
                                         TQDMProgressBar, EarlyStopping)
from pytorch_lightning.loggers import WandbLogger
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import CocoCaptions
from transformers import (
    AutoTokenizer,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
)
import submitit
from torchmetrics.functional import bleu_score

# Local application/library specific imports
from adapter_utils import AdapterConf, PartialFTConf, RNNAdapterConf, adapter_init, rnn_adapter_init
from adapter_utils import FTConf
import random
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
del os.environ["LD_LIBRARY_PATH"]


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AvgMetric(Metric):
    """
    A callable that updates metrics across training batches
    and computes averaged value for recorded metrics.
    """

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        # pyre-fixme[2]
        process_group: Optional[Any] = None,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
        )

        # The way PyTorch Lightning Metrics manage state does not work with pyre.
        self.add_state(
            "curr_val",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "num_examples",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )

    # pyre-fixme[14]: `update` overrides method defined in `Metric` inconsistently.
    def update(self, val: torch.Tensor, num_samples: torch.Tensor) -> None:

        # pyre-fixme[29]
        # pyre-ignore
        self.curr_val += val
        # pyre-fixme[29]
        # pyre-ignore
        self.num_examples += num_samples

    def compute(self) -> torch.Tensor:
        # pyre-fixme[29]
        if self.num_examples.item() == 0.0:
            # avg loss defaults to -1 if no samples detected;
            # pyre-fixme[6]
            return -torch.ones_like(self.num_examples)
        # pyre-fixme[29]
        return self.curr_val / self.num_examples


class Config(NamedTuple):
    learning_rate: float = 1e-4
    peft_conf: FTConf = FTConf()
    max_epochs: int = 20


def collate_fn(data):
    batch = {}
    for key in data[0]:
        batch[key] = ([d[key] for d in data] if key != "pixel_values" else
                      torch.cat([d[key] for d in data]))
    return batch


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
        max_length=25,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
            vision_model_name_or_path, text_model_name_or_path)
        self._freeze(self.model.encoder)
        self.peft_conf = peft_conf
        self.setup_ft()

        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.data_module = data_module
        self.data_module.setup()

        self.learning_rate = learning_rate
        self.max_length = max_length

        self.val_loss = AvgMetric()
        self.test_loss = AvgMetric()

        self.eval_results = []
        self.test_results = []

    def setup_ft(self) -> None:
        num_orig_params = sum(p.numel() for p in self.model.parameters())

        if self.peft_conf.ft_type == "full":
            self.enable_full_tuning(num_orig_params)
        elif self.peft_conf.ft_type == "partial":
            self.enable_partial_tuning(num_orig_params)
        elif self.peft_conf.ft_type == "adapter":
            self.enable_adapter_tuning(num_orig_params)
        elif self.peft_conf.ft_type == "rnn_adapter":
            self.enable_rnn_adapter_tuning(num_orig_params)
        else:
            raise ValueError(f"Unsupported FT type {self.peft_conf.ft_type}")

    def enable_full_tuning(self, num_orig_params: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = True
        print(f"Enables full tuning; num trainable params {num_orig_params:,}")

    def enable_partial_tuning(self, num_orig_params: int) -> None:
        p_num = 0
        for n, p in self.model.named_parameters():
            if self.peft_conf.ft_key in n:
                p.requires_grad = True
                p_num += p.numel()
            else:
                p.requires_grad = False
        ratio = round(p_num / num_orig_params, 3)
        print(
            f"Enables partial tuning on {self.peft_conf.ft_key}; num trainable params {p_num:,}, ratio {ratio}"
        )

    def enable_adapter_tuning(self, num_orig_params: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        encoder_adapters = adapter_init(
            model=self.model.encoder,
            layers=self.model.encoder.encoder.layer,
            adapter_conf=self.peft_conf,
            std=None,
            register_hooks=True,
        )
        decoder_adapters = adapter_init(
            model=self.model.decoder,
            layers=self.model.decoder.bert.encoder.layer,
            adapter_conf=self.peft_conf,
            std=None,
            register_hooks=True,
        )
        encoder_p_num = sum(p.numel() for adapter in encoder_adapters
                            for p in adapter.parameters())
        decoder_p_num = sum(p.numel() for adapter in decoder_adapters
                            for p in adapter.parameters())
        p_num = encoder_p_num + decoder_p_num
        ratio = round(p_num / num_orig_params, 3)
        print(
            f"Enables adapter tuning; num trainable params {p_num:,}, ratio {ratio}"
        )

    def enable_rnn_adapter_tuning(self, num_orig_params: int) -> None:
        for p in self.model.parameters():
            p.requires_grad = False

        encoder_rnn_adapter = rnn_adapter_init(
            model=self.model.encoder,
            layers=self.model.encoder.encoder.layer,
            merge_module=self.model.encoder.layernorm,
            rnn_adapter_conf=self.peft_conf,
            register_hooks=True,
        )
        decoder_rnn_adapter = rnn_adapter_init(
            model=self.model.decoder,
            layers=self.model.decoder.bert.encoder.layer,
            merge_module=self.model.decoder.bert.encoder.layer[11].output.
            LayerNorm,
            rnn_adapter_conf=self.peft_conf,
            register_hooks=True,
        )
        encoder_p_num = sum(p.numel()
                            for p in encoder_rnn_adapter.parameters())
        decoder_p_num = sum(p.numel()
                            for p in decoder_rnn_adapter.parameters())
        p_num = encoder_p_num + decoder_p_num
        ratio = round(p_num / num_orig_params, 3)
        print(
            f"Enables rnn-adapter tuning; num trainable params {p_num:,}, ratio {ratio}"
        )

    def _freeze(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def _evaluate(self, results):
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
        annotation_file = '/data/home/ngjhn/read/coco/annotations/captions_val2014.json'

        # create coco object and coco_result object
        coco = COCO(annotation_file)
        coco_result = coco.loadRes(results)

        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco, coco_result)

        # evaluate on a subset of images by setting
        # coco_eval.params['image_id'] = coco_result.getImgIds()
        # please remove this line when evaluating the full validation set
        coco_eval.params['image_id'] = coco_result.getImgIds()

        # evaluate results
        # SPICE will take a few minutes the first time, but speeds up due to caching
        coco_eval.evaluate()

        # print output evaluation scores
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
        return coco_eval.eval

    def evaluate(self, device):
        all_results = []
        with torch.no_grad():
            self.model.eval()
            self.model.to(device)
            for batch in tqdm(self.data_module.test_dataloader()):
                images, image_ids = batch['pixel_values'], batch['image_id']
                images = images.to(device)

                generated_ids = self.model.generate(images)
                generated_text = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True)
                results = [{
                    "image_id": image_id,
                    "caption": caption
                } for image_id, caption in zip(image_ids, generated_text)]
                all_results.extend(results)
        return self._evaluate(all_results)

    def forward(self, images, labels):
        return self.model(pixel_values=images, labels=labels)

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
        captions = batch['captions']
        labels = self.pad_text(captions).to(self.device)
        output = self.model(pixel_values=images, labels=labels)
        if batch_idx == 0:
            results = self.evaluate(self.device)
            self.log_dict(results)

        self.val_loss.update(output.loss.item(), 1)

    def test_step(self, batch, batch_idx):
        images = batch["pixel_values"].to(self.device)
        captions = batch['captions']
        labels = self.pad_text(captions).to(self.device)
        output = self.model(pixel_values=images, labels=labels)
        self.test_loss.update(output.loss.item(), 1)

    def on_validation_epoch_end(self) -> None:
        self.log("val_loss", self.val_loss.compute())
        self.val_loss.reset()

    def on_test_epoch_end(self) -> None:
        self.log("test_loss", self.test_loss.compute())
        self.test_loss.reset()

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.learning_rate)

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


def sweep_iteration(config=None):
    print("Starting sweep iteration")
    seed_everything(1)
    torch.set_float32_matmul_precision('high')
    with wandb.init():
        peft_confg = RNNAdapterConf(
            input_dim=768,
            rnn_dim=64,
            num_layers=1,
            rnn_type="gru",
        )
        name = peft_confg.ft_type

        config = wandb.config if config is None else config
        wandb_logger = WandbLogger()
        data_module = CocoDataModule(
            data_dir="/data/home/ngjhn/read/coco",
            ann_dir="/data/home/ngjhn/read/coco/annotations/",
            vision_model_name_or_path="google/vit-base-patch16-224-in21k",
            split_file="/data/home/ngjhn/read/coco/dataset_coco.json",
            collate_fn=collate_fn,
            train_batch_size=32,
            val_batch_size=128)

        model = ImageCaptioningModel(
            "google/vit-base-patch16-224-in21k",
            "bert-base-uncased",
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
                    filename=f'{name}' + f"-lr:{config.learning_rate:.4f}" +
                    '-{epoch}-{val_loss:.2f}',
                    dirpath='/data/home/ngjhn/read/lightning_logs/full',
                    auto_insert_metric_name=True),
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
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
    # sweep_id = wandb.sweep(sweep_config, project='read')
    wandb.agent("ry0ss5vc", function=sweep_iteration, count=10)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Example script with SLURM arguments")
    parser.add_argument("--slurm_folder",
                        type=str,
                        required=True,
                        help="Path to the SLURM folder")
    parser.add_argument("--name",
                        type=str,
                        required=True,
                        help="Name of the experiment")
    parser.add_argument("--ft_type",
                        type=str,
                        required=True,
                        help="Name of FT type")
    parser.add_argument("--learning_rate",
                        type=float,
                        required=True,
                        help="Learning rate for the model")
    parser.add_argument("--sweep",
                        action="store_true",
                        help="Enable sweep mode (default: False)")

    args = parser.parse_args()
    # Submit a job to the SLURM queue, batch mode only. Find logs
    # in the conf.slurm_folder.
    gpus_per_node = 1
    nodes = 1
    if args.sweep:
        executor = submitit.AutoExecutor(folder=args.slurm_folder)
        executor.update_parameters(
            name=f"{args.name}",
            slurm_partition='lowpri',
            nodes=nodes,
            tasks_per_node=1,
            slurm_gpus_per_task=1,
            cpus_per_task=10 * gpus_per_node,
            slurm_time=1000,
        )
        sweep_config = {
            "method": "bayes",
            "name": args.name,
            "metric": {
                "name": "val_loss",
                "goal": "maximize"
            },
            "early_terminate": {
                "type": "hyperband",
                "min_iter": 10
            },
            "parameters": {
                "learning_rate": {
                    "min": 1e-6,
                    "max": 1e-3
                },
            },
        }
        with executor.batch():
            job = executor.submit(sweep, sweep_config)
    else:
        executor = submitit.AutoExecutor(folder=args.slurm_folder)
        executor.update_parameters(
            name=f"{args.name}",
            slurm_partition='lowpri',
            nodes=nodes,
            tasks_per_node=1,
            slurm_gpus_per_task=1,
            cpus_per_task=10 * gpus_per_node,
            slurm_time=1000,
        )
        # executor is the submission interface (logs are dumped in the folder)
        executor = submitit.AutoExecutor(folder=args.slurm_folder)
        # set timeout in min, and partition for running the job
        executor.update_parameters(
            timeout_min=1000,
            slurm_partition="lowpri",
            name=f"{args.name}",
            slurm_gres="gpu:1",
        )
        if args.ft_type == "full":
            peft_confg = FTConf(ft_type="full")
        elif args.ft_type == "partial":
            peft_confg = FTConf(ft_type="partial")
        elif args.ft_type == "bias":
            peft_confg = FTConf(ft_type="bias")
        elif args.ft_type == "adapters":
            peft_confg = AdapterConf(
                input_dim=768,
                hidden_dim=64,
                output_dim=768,
            )
        elif args.ft_type == "rnn":
            peft_confg = RNNAdapterConf(
                input_dim=768,
                rnn_dim=64,
                num_layers=1,
                rnn_type="gru",
            )

        with executor.batch():
            job = executor.submit(
                sweep_iteration,
                Config(learning_rate=args.learning_rate, peft_conf=peft_confg))
        print(f"Submitted job {job.job_id} to SLURM queue")
