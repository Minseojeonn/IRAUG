import torch.utils.data as data
import numpy as np

class DatasetClass(data.Dataset):
    """
    Dataset class.
    
    args:
        edge (np.array): edge
        label (np.array): label
    
    return 
        dataset
    """
    def __init__(self, edge: np.array, label: np.array) -> None:
        self.edge = edge
        self.label = label
        self.len = edge.shape[0]
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        return self.edge[index], self.label[index]
    

