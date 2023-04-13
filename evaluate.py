from coco import *
from adapter_utils import *
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import os

ft_types = ["adapter", "bias", "full", "rnn", "rnn_adapter"]
best_files = []
for ft_type in ft_types:
    base_dir = f"/data/home/ngjhn/read/lightning_logs/{ft_type}"
    files = os.listdir(base_dir)
    if files:
        file = sorted([(file.split("val_loss=")[1], file) for file in files],
                      key=lambda x: x[0])[0][1]
        best_files.append(os.path.join(base_dir, file))

data_module = CocoDataModule(
    data_dir="/data/home/ngjhn/read/coco",
    ann_dir="/data/home/ngjhn/read/coco/annotations/",
    vision_model_name_or_path="google/vit-base-patch16-224-in21k",
    split_file="/data/home/ngjhn/read/coco/dataset_coco.json",
    collate_fn=collate_fn,
    train_batch_size=32,
    val_batch_size=128)

for best_file in best_files:
    model = ImageCaptioningModel(
        "google/vit-base-patch16-224-in21k",
        "bert-base-uncased",
        data_module=data_module,
        learning_rate=0.1,
        peft_conf= FTConf(ft_type="full"),
    )
    model = model.load_from_checkpoint(
        best_file,
        vision_model_name_or_path="google/vit-base-patch16-224-in21k",
        text_model_name_or_path="bert-base-uncased",
        data_module=data_module,
        learning_rate=0.1,
        peft_conf= FTConf(ft_type="full"),
    )
    print(best_file)
    model.evaluate("cuda:0")
