import os
import pandas as pd
from tqdm import tqdm
from PIL import Image
from emotion_detector.datasets.dataset import Dataset
from emotion_detector.datasets.label_reader import LabelReader
from transformers import AutoImageProcessor
from datasets import Dataset as HFDataset

class FileClassification(Dataset):
    def __init__(self, data_dir: str, file: str, path_key: str= "file", label_key: str = "label"):
        super().__init__()

        self.data_dir = data_dir
        self.file = os.path.join(data_dir, file)
        self.path_key = path_key
        self.label_key = label_key

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            "filepath" : self.full_paths[idx],
            f"{self.path_key}" : self.relative_paths[idx],
            f"{self.label_key}" : self.labels[idx]
        }
    
    def _build(self):
        df = pd.read_csv(self.file)

        relative_paths = df[self.path_key].tolist()
        full_paths = df[self.path_key].map(lambda x: os.path.join(self.data_dir, x)).tolist()
        labels = df[self.label_key].tolist()

        self.relative_paths = relative_paths
        self.full_paths = full_paths
        self.labels = labels


class ImageFileClassificationDataset(FileClassification):
    def __init__(
            self,
            data_dir: str,
            file: str,
            label_reader: LabelReader,
            img_processor: AutoImageProcessor,
            path_key: str = "file",
            label_key: str = "label",
            num_samples = -1
    ):
        self.label_reader = label_reader
        self.img_processor = img_processor
        self.num_samples = num_samples

        super().__init__(
            data_dir=data_dir,
            file=file,
            path_key=path_key,
            label_key=label_key
        )
        self.build()

    def to_hf_dataset(self):
        def transform(example_batch):
            inputs = self.img_processor([x for x in example_batch["images"]], return_tensors="pt")
            inputs["labels"] = example_batch["labels"]

            return inputs
        
        data_dict = {
            "images" : [],
            "labels": []
        }

        for data in tqdm(self):
            filepath = data["filepath"]
            image = Image.open(filepath)
            image = Image.convert("RGB")
            data_dict["images"].append(image)

            label = data[self.label_key]
            label_idx = self.label_reader.get_label_idx(label)
            data_dict["labels"].append(label_idx)

            if self.num_samples > 0 and len(data_dict["images"]) >= self.num_samples:
                break 
        
        dataset = HFDataset.from_dict(data_dict)
        dataset = dataset.with_transform(transform)

        return dataset
