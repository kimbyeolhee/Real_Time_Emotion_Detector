import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from emotion_detector.datasets.label_reader import LabelReader
from emotion_detector.datasets.image_file_classification import ImageFileClassificationDataset

import torch
from datasets import load_metric
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import TrainingArguments, Trainer


def main(config):
    label_file = os.path.join(config.root_path, config.data.folder_path, config.data.labels_file_name)
    label_reader = LabelReader(label_file)
    labels = label_reader.get_labels()

    print("Loading dataset ---")
    train_file = os.path.join(config.root_path, config.data.folder_path, config.data.train_file_name)
    eval_file = os.path.join(config.root_path, config.data.folder_path, config.data.eval_file_name)
    
    processor = ViTImageProcessor.from_pretrained(config.model.name_or_path)
    train_ds = ImageFileClassificationDataset(
        data_dir = os.path.join(config.root_path, config.data.folder_path),
        file = train_file,
        label_reader = label_reader,
        img_processor = processor
    )
    train_dataset = train_ds.to_hf_dataset() # consist of 'pixel_values', 'label' keys
    # print(train_dataset[0]["pixel_values"].shape)

    eval_ds = ImageFileClassificationDataset(
        data_dir = os.path.join(config.root_path, config.data.folder_path),
        file = eval_file,
        label_reader = label_reader,
        img_processor = processor
    )
    eval_dataset = eval_ds.to_hf_dataset()

    model = ViTForImageClassification.from_pretrained(
        config.model.name_or_path,
        num_labels=len(labels),
        id2label={str(i) : c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)}
        )
    
    training_args = TrainingArguments(
        output_dir = config.training_args.output_dir,
        per_device_train_batch_size = config.training_args.per_device_train_batch_size,
        evaluation_strategy = config.training_args.evaluation_strategy,
        num_train_epochs = config.training_args.num_train_epochs,
        use_cpu = config.training_args.use_cpu,
        save_steps = config.training_args.save_steps,
        eval_steps = config.training_args.eval_steps,
        logging_steps = config.training_args.logging_steps,
        learning_rate = config.training_args.learning_rate,
        save_total_limit = config.training_args.save_total_limit,
        load_best_model_at_end = config.training_args.load_best_model_at_end,
        report_to = config.training_args.report_to,
        remove_unused_columns = config.training_args.remove_unused_columns
    )

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "labels": torch.tensor([x["label"] for x in batch])
        }
    
    metric = load_metric("accuracy")
    def compute_metrics(p):
        return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
    
    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = collate_fn,
        compute_metrics = compute_metrics,
        tokenizer = processor
    )

    print("Training models ---")
    train_results = trainer.train()

    trainer.save_model()



if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", "-c", type=str, default="base_config", help="config file path")
    args, _ = parser.parse_known_args()
    config = OmegaConf.load(f"./configs/{args.config}.yaml")

    main(config)

