import os
import argparse
from omegaconf import OmegaConf
from emotion_detector.datasets.label_reader import LabelReader
from emotion_detector.datasets.image_file_classification import ImageFileClassificationDataset

from transformers import ViTImageProcessor

def main(config):
    label_file = os.path.join(config.root_path, "data/labels.txt")
    label_reader = LabelReader(label_file)
    labels = label_reader.get_labels()

    print("Loading dataset ---")
    train_file = os.path.join(config.root_path, config.data.folder_path, config.data.train_file_name)

    processor = ViTImageProcessor.from_pretrained(config.model.name_or_path)
    dataset = ImageFileClassificationDataset(
        data_dir = os.path.join(config.root_path, config.data.folder_path),
        file = train_file,
        label_reader = label_reader,
        img_processor = processor
    )
    train_dataset = dataset.to_hf_dataset()

    print(train_dataset[0])


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config", help="config file path")
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    main(config)

