# template for data loader
import numpy as np
import torch.utils.data.dataset as Dataset
import torch

from dataset.utils import split_data, load_data
from dataset.DatasetClass import DatasetClass


class DataTemplate(object):
    """Template for data loader
        for unsigned graph

    Args:
        model (str): Model
        dataset_name (str): dataset name
        seed (int): seed
        split_ratio (list): [train(float), val(float), test(float)], train+val+test == 1 
        dataset_shuffle (bool): dataset_shuffle if True
        device (str): device
        direction (str): True-direct, False-undirect
        node_idx_type (str): "uni" or "bi"
    """

    def __init__(
        self,
        dataset_name: str,
        seed: int,
        split_ratio: list,
        dataset_shuffle: bool,
        device: str,
        direction: bool,
        node_idx_type: str,
        input_dim: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = f"./dataset/{self.dataset_name}.tsv"
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset_shuffle = dataset_shuffle
        self.device = device
        self.direction = direction
        self.node_idx_type = node_idx_type
        self.input_dim = input_dim
        assert node_idx_type.lower() in [
            "uni", "bi"], "not supported node_idx_type"
        assert np.isclose(sum(split_ratio), 1).item(
        ), "sum of split_ratio is not 1"
        self.processing()

    def processing(
        self,
    ):
        array_of_edges, self.num_nodes, self.num_edges = load_data(
            self.dataset_path, self.direction, self.node_idx_type)
        processed_dataset = split_data(
            array_of_edges, self.split_ratio, self.seed, self.dataset_shuffle)
        processed_dataset["init_emb"] = self.set_init_embeddings() 
        self.processed_dataset = processed_dataset

    def get_dataset(self):
        train_dataset = DatasetClass(self.processed_dataset["train_edges"], self.processed_dataset["train_label"])
        val_dataset = DatasetClass(self.processed_dataset["valid_edges"], self.processed_dataset["valid_label"])
        test_dataset = DatasetClass(self.processed_dataset["test_edges"], self.processed_dataset["test_label"]) 
        
        return train_dataset, val_dataset, test_dataset
    
    def set_init_embeddings(self):
        """
        set embeddings function for training model

        Args:
            embeddings (torch.Tensor): embeddings
        """
        if self.node_idx_type == "uni":
            embeddings = torch.nn.init.xavier_uniform_(torch.empty(
                (sum(self.num_nodes), self.input_dim)))
            return embeddings
        elif self.node_idx_type == "bi":
            self.embeddings_user = torch.nn.init.xavier_uniform_(
                torch.empty(self.num_nodes[0], self.input_dim))
            self.embeddings_item = torch.nn.init.xavier_uniform_(
                torch.empty(self.num_nodes[1], self.input_dim))
            return [self.embeddings_user, self.embeddings_item]