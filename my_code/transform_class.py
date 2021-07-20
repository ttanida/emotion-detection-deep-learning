from torch.utils.data import Dataset


class ApplyTransform(Dataset):
    """
    Apply transformations to a Dataset

    Arguments:
        dataset (Dataset): A Dataset that returns (sample, target)
        transform (callable): A function/transform to be applied on the sample
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        sample, target = self.dataset[idx]
        sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.dataset)
        
