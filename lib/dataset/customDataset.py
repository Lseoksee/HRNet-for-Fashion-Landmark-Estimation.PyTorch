from torch.utils.data.dataset import ConcatDataset
from .JointsDataset import JointsDataset


class CustomDataset(JointsDataset):
    
    def __init__(self) -> None:
        pass
        
        
    def __getitem__(self, index):
        return super().__getitem__(index)

    
    def __len__():  
        pass