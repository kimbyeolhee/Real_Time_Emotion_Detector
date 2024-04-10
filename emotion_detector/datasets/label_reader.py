class LabelReader:
    def __init__(self, labels_file=None):
        self.labels_files = labels_file
        self.label_indices = {}
        self._build()

    def _build(self):
        print("Loading labels ---")
        if self.labels_files is None:
            raise Exception("labels_file is not provided")
        else:
            with open(self.labels_files, "r") as f:
                self.class_labels = f.read().splitlines()
                for i, label in enumerate(self.class_labels):
                    self.label_indices[i] = label
            print(f"{len(self.class_labels)} labels loaded")
    
    def num_labels(self):
        return len(self.class_labels)
    
    def idx_to_label(self, idx):
        return self.class_labels[idx]
    
    def label_to_idx(self, label):
        return self.label_indices[label]
    
    def get_labels(self):
        return self.class_labels
