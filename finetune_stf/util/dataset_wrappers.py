from torch.utils.data import Dataset


class RepeatDataset(Dataset):
    def __init__(self, dataset, repeat):
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}")
        self.dataset = dataset
        self.repeat = int(repeat)

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

