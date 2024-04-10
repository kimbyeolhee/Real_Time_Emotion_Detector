class Dataset:
    def __init__(self):
        pass

    def __len__(self):
        raise NotImplementedError("must implement __len__ method")
    
    def __getitem__(self, idx):
        raise NotImplementedError("must implement __getitem__ method")
    
    def _build(self):
        raise NotImplementedError("must implement _build method")
    
    def build(self):
        try:
            self.build()
        except NotImplementedError:
            print("must implement _build() method")