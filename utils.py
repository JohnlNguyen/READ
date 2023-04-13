import argparse
from torchmetrics.metric import Metric

import os
from typing import NamedTuple, Optional, Any

# Third-party imports
import pytorch_lightning as pl
import torch

# Local application/library specific imports
from adapter_utils import *
from utils import *
import random
import numpy as np


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


def evaluate_coco(results):
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


def parse_args():
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
    parser.add_argument("--sweep_id",
                        type=str,
                        required=False,
                        help="Sweep ID",
                        default="")
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
    parser.add_argument("--hidden_dim",
                        type=int,
                        required=False,
                        help="Adapter Hidden Dimension",
                        default=256)
    parser.add_argument("--scale_factor",
                        type=float,
                        required=False,
                        help="Scale Factor",
                        default=1.0)
    parser.add_argument('--bidirectional',
                        dest='bidirectional',
                        action='store_true')
    parser.add_argument("--bottle_neck_ratio",
                        type=int,
                        required=False,
                        help="Lora Ratio",
                        default=8)
    parser.add_argument("--alpha",
                        type=int,
                        required=False,
                        help="Lora Alpha",
                        default=8)
    return parser.parse_args()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
