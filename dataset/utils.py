import torch
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
import torch.utils.data as data


def split_data(
    user_item_dict: dict,
    split_ratio: list,
    seed: int,
    dataset_shuffle: bool,
) -> dict:
    """Split your dataset into train, valid, and test

    Args:
        array_of_edges (np.array): array_of_edges
        split_ratio (list): train:test:val = [float, float, float], train+test+val = 1.0 
        seed (int) = seed
        dataset_shuffle (bool) = shuffle dataset when split

    Returns:
        dataset_dict: {train_edges : np.array, train_label : np.array, test_edges: np.array, test_labels: np.array, valid_edges: np.array, valid_labels: np.array}
    """

    assert np.isclose(sum(split_ratio), 1), "train+test+valid != 1"
    train_ratio, valid_ratio, test_ratio = split_ratio
    
    num_user = len(user_item_dict)
    num_train = int(num_user * train_ratio)
    num_valid = int(num_user * valid_ratio)
    num_test = num_user - num_train - num_valid
    
    user_idx = [i for i in range(num_user+1)]
    user_idx = np.random.permutation(user_idx).tolist()
    
    train = {}
    valid = {}
    test = {}
    
    for i in user_idx[:num_train]:
        train[i] = user_item_dict[i]
    for i in user_idx[num_train:num_train+num_valid]:
        valid[i] = user_item_dict[i]
    for i in user_idx[num_train+num_valid:]:
        test[i] = user_item_dict[i]
    
    proprecessed_dataset = {"train": train, "valid": valid, "test": test}
    
    return proprecessed_dataset


def load_data(
    dataset_path: str,
    direction: bool,
) -> np.array:
    """Read data from a file

    Args:
        dataset_path (str): dataset_path
        direction (bool): True=direct, False=undirect
        node_idx_type (str): "uni" - no intersection with [uid, iid], "bi" - [uid, iid] idx has intersection

    Return:
        array_of_edges (array): np.array of edges
        num_of_nodes: [type1(int), type2(int)]
    """

    edgelist = []
    with open(dataset_path) as f:
        for line in f:
            a, b, s = map(int, line.split('\t'))
            if s == -1:
                s = 0
            edgelist.append((a, b, s))
    num_of_nodes = get_num_nodes(np.array(edgelist))
    edgelist = np.array(edgelist)

    assert max(edgelist[:,0]) > min(edgelist[:,1]), "user id and item id must be separated"
    edgelist[:,1] = edgelist[:,1] + num_of_nodes[0]
    if direction == False:
        edgelist = edgelist.tolist()
        for idx, edgelist in enumerate(edgelist):
            fr, to, sign = edgelist
            edgelist.append(to, fr, sign)
        edgelist = np.array(edgelist)

    num_edges = np.array(edgelist).shape[0]
    
    user_item_dict = defaultdict(list)
    for edge in edgelist:
        user_item_dict[edge[0]].append(edge[1])
        
    return edgelist, num_of_nodes, num_edges, user_item_dict


def get_num_nodes(
    dataset: np.array
) -> int:
    """get num nodes when bipartite

    Args:
        dataset (np.array): dataset

    Returns:
        num_nodes tuple(int, int): num_nodes_user, num_nodes_item
    """
    num_nodes_user = np.amax(dataset[:, 0]) + 1
    num_nodes_item = np.amax(dataset[:, 1]) + 1
    return (num_nodes_user.item(), num_nodes_item.item())


def collate_fn(batch):
    breakpoint()
    user, items = zip(*batch)
    return user, items