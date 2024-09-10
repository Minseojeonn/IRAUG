# template for data loader
import numpy as np
import torch.utils.data.dataset as Dataset
import torch

from dataset.utils import split_data, load_data
from dataset.DatasetClass import TrnDatasetClass, EvalDatasetClass


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
    """

    def __init__(
        self,
        dataset_name: str,
        seed: int,
        split_ratio: list,
        dataset_shuffle: bool,
        device: str,
        direction: bool,
        input_dim: int,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_path = f"./dataset/{self.dataset_name}.tsv"
        self.seed = seed
        self.split_ratio = split_ratio
        self.dataset_shuffle = dataset_shuffle
        self.device = device
        self.direction = direction
        self.input_dim = input_dim
        assert np.isclose(sum(split_ratio), 1).item(
        ), "sum of split_ratio is not 1"
        self.processing()
        self.build_trainnormajd()
        

    def processing(
        self,
    ):
        array_of_edges, self.num_nodes, self.num_edges = load_data(
            self.dataset_path, self.direction)
        processed_dataset = split_data(
           array_of_edges ,self.split_ratio, self.seed, self.dataset_shuffle)
        processed_dataset["init_emb"] = self.set_init_embeddings() 
        self.processed_dataset = processed_dataset
        

    def get_dataset(self):
        train_dataset = TrnDatasetClass(self.processed_dataset["train_edges"], self.processed_dataset["train_label"], self.num_nodes)
        val_dataset = EvalDatasetClass(self.processed_dataset["valid_edges"], self.processed_dataset["valid_label"], self.num_nodes)
        test_dataset = EvalDatasetClass(self.processed_dataset["test_edges"], self.processed_dataset["test_label"], self.num_nodes) 
        
        return train_dataset, val_dataset, test_dataset, self.num_nodes
    
    def set_init_embeddings(self):
        """
        set embeddings function for training model

        Args:
            embeddings (torch.Tensor): embeddings
        """
        self.embeddings_user = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_nodes[0], self.input_dim))
        self.embeddings_item = torch.nn.init.xavier_uniform_(
            torch.empty(self.num_nodes[1], self.input_dim))
        return [self.embeddings_user, self.embeddings_item]
        
    def build_trainnormajd(self):
        self.adj_matrix = torch.sparse_coo_tensor(torch.LongTensor(self.processed_dataset["train_edges"]).T, torch.LongTensor(self.processed_dataset["train_label"]), torch.Size([sum(self.num_nodes), sum(self.num_nodes)]), dtype=torch.long, device="cpu")
        #dense = self.adj_matrix.to_dense().abs().float()
        row_sum = torch.sum(self.adj_matrix.abs(), dim=1).float() #row sum
        col_sum = torch.sum(self.adj_matrix.abs(), dim=0).float() #col sum
        
        d_inv_row = torch.pow(row_sum, -0.5).flatten()
        d_inv_col = torch.pow(col_sum, -0.5).flatten()
        
        del row_sum, col_sum
        
        d_mat_row = torch.diag(d_inv_row.to_dense()).to_sparse()
        d_mat_col = torch.diag(d_inv_col.to_dense()).to_sparse()
        
        del d_inv_row, d_inv_col
        
        norm_adj = d_mat_row @ self.adj_matrix 
        norm_adj = norm_adj @ d_mat_col
        norm_adj = norm_adj.to_sparse()
        self.adj_matrix = norm_adj    
        del d_mat_row, d_mat_col, norm_adj 
    
    def get_adj_matrix(self):
        return self.adj_matrix
    
    def get_embeddings(self):
        return self.processed_dataset["init_emb"]