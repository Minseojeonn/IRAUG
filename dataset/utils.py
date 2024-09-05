import torch
from collections import defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
import torch.utils.data as data


def split_data(
    array_of_edges: np.array,
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
    train_X, test_val_X, train_Y, test_val_Y = train_test_split(
        array_of_edges[:, :2], array_of_edges[:, 2], test_size=1 - train_ratio, random_state=seed, shuffle=dataset_shuffle)
    val_X, test_X, val_Y, test_Y = train_test_split(test_val_X, test_val_Y, test_size=test_ratio/(
        test_ratio + valid_ratio), random_state=seed, shuffle=dataset_shuffle)

    dataset_dict = {
        "train_edges": train_X,
        "train_label": train_Y,
        "valid_edges": val_X,
        "valid_label": val_Y,
        "test_edges": test_X,
        "test_label": test_Y
    }

    return dataset_dict


def load_data(
    dataset_path: str,
    direction: bool,
    node_idx_type: str
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

    if node_idx_type.lower() == "uni":
        for idx, edge in enumerate(edgelist.tolist()):
            fr, to, sign = edge
            edgelist[idx] = (fr, to+num_of_nodes[0], sign)
        edgelist = np.array(edgelist)
        assert len(set(edgelist[:, 0].tolist()).intersection(
            set(edgelist[:, 1].tolist()))) == 0, "something worng"

    if direction == False:
        edgelist = edgelist.tolist()
        for idx, edgelist in enumerate(edgelist):
            fr, to, sign = edgelist
            edgelist.append(to, fr, sign)
        edgelist = np.array(edgelist)

    num_edges = np.array(edgelist).shape[0]

    if node_idx_type.lower() == "bi" and direction == False:
        raise Exception("undirect can not use with bi type.")

    return edgelist, num_of_nodes, num_edges


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


def split_edges_by_sign(
    edge_lists: np.array,
    edge_label: np.array,
    consider_direction: bool
) -> list:
    """split edge list by sign

    Args:
        edge_lists (np.array): edge_array
        edge_label (np.array): edge_sign
        consider_direction (bool): 
            False : return edgelist_pos, edgelist_neg
            True : return edgelist_pos_u_i, edgelist_neg_u_i, edgelist_i_u, edgelist_i_u

    Returns:
        pos_edges (list): pos_edges
        neg_edges (list): neg_edges
    """

    edgelist_pos_u_i, edgelist_neg_u_i = [[], []], [[], []]
    edgelist_pos_i_u, edgelist_neg_i_u = [[], []], [[], []]

    for (fr, to), sign in zip(edge_lists.tolist(), edge_label.tolist()):
        if sign == 1:
            edgelist_pos_u_i[0].append(fr)
            edgelist_pos_u_i[1].append(to)
            edgelist_pos_i_u[0].append(to)
            edgelist_pos_i_u[1].append(fr)

        elif sign == 0:
            edgelist_neg_u_i[0].append(fr)
            edgelist_neg_u_i[1].append(to)
            edgelist_neg_i_u[0].append(to)
            edgelist_neg_i_u[1].append(fr)

        else:
            print(fr, to, sign)
            raise Exception("sign must be 0/1")
    if consider_direction:
        return edgelist_pos_u_i, edgelist_neg_u_i, edgelist_pos_i_u, edgelist_neg_i_u
    else:
        return edgelist_pos_u_i, edgelist_neg_u_i


def make_degree_inv(
    R: torch.sparse_coo_tensor
) -> torch.tensor:
    """ get degree inverse

    Args:
        R (torch.sparse_coo_tensor): bi-adjacency matrix

    Returns:
        u_diag_inv_list (tnesor): inversed degree matrix col
        v_diag_int_list (tensor): inversed degree matrix row
    """

    abs_R = abs(R)
    u_diag_tensor = torch.sum(abs_R, dim=1).to_dense()
    v_diag_tensor = torch.sum(abs_R, dim=0).to_dense()

    u_diag_tensor[u_diag_tensor == 0] = 1
    v_diag_tensor[v_diag_tensor == 0] = 1

    u_diag_inv_list = 1.0 / u_diag_tensor
    v_diag_int_list = 1.0 / v_diag_tensor

    return u_diag_inv_list, v_diag_int_list


def normalization(
    device: str,
    R_pos: torch.Tensor,
    R_neg: torch.Tensor,
    d_u_inv: torch.Tensor,
    d_v_inv: torch.Tensor
) -> torch.Tensor:
    """ normalization

    Args:
        R_pos (Tensor): bi-adjacency matrix related to positive link
        R_neg (Tensor): bi-adjacency matrix related to negative link
        d_u_inv (Tensor): degree matrix about node type "u" - row
        d_v_inv (Tensor): degree matrix about node type "v" - col

    Returns:
        A_pos (Tensor): normalized matrix of posivtve about u to v
        A_neg (Tensor): normalized matrix of negative about u to v
        B_pos (Tensor): normalized matrix of posivtve about v to u
        B_neg (Tensor): normalized matrix of negative about v to u
    """

    R_pos = R_pos.to(device)
    R_neg = R_neg.to(device)
    d_u_inv = d_u_inv.to(device)
    d_v_inv = d_v_inv.to(device)
    d_u_inv = torch.diag(d_u_inv)
    d_v_inv = torch.diag(d_v_inv)
    A_pos = torch.mm(d_u_inv, R_pos).to_sparse()
    A_neg = torch.mm(d_u_inv, R_neg).to_sparse()

    B_pos = torch.mm(d_v_inv, R_pos.t()).to_sparse()
    B_neg = torch.mm(d_v_inv, R_neg.t()).to_sparse()

    return A_pos.to("cpu"), A_neg.to("cpu"), B_pos.to("cpu"), B_neg.to("cpu")


def svd(
    device: str,
    normalized_pos_matrix: torch.tensor,
    normalized_neg_matrix: torch.tensor,
    rank: int
):
    """svd

    Args:
        M_pos (Tensor.sparse): pos adj matrix 
        M_neg (Tensor.sparse): neg adj matrix
        rank (int): SVD Rank

    Returns:
        (dict): stored dictionary about element of SVD\
            from two given(it stored to 2 cases(pos/neg) 6 elements) 
    """
    normalized_pos_matrix = normalized_pos_matrix.to(device)
    normalized_neg_matrix = normalized_neg_matrix.to(device)
    store_dict = dict()
    M = [normalized_pos_matrix, normalized_neg_matrix]
    
    key_usv_list = ['u', 's', 'v']
    key_sign_list = ['pos', 'neg']

    key_list = [key_usv_list[j] + '_' + key_sign_list[i]
                for i in range(len(key_sign_list))
                for j in range(len(key_usv_list))]

    for i in range(len(M)):
        U, S, V = torch.svd_lowrank(M[i].coalesce(), rank)

        store_dict[key_list[0+(3*i)]] = U.to("cpu")
        store_dict[key_list[1+(3*i)]] = S.to("cpu")
        store_dict[key_list[2+(3*i)]] = V.to("cpu")

    return store_dict


def generate_mask(
    row: int,
    column: int,
    masked_ratio: float
):
    """generate mask

    Args:
        row (int): shape[0]
        column (int): shape[1]
        masked_ratio (float): masked ratio 0<ratio<=1

    Returns:
        arr_mask (np.array): shape=(row,col)
    """
    assert masked_ratio > 0 and masked_ratio <= 1, "masked ratio between 0 and 1"

    # 1 -- leave   0 -- drop
    arr_mask_ratio = np.random.uniform(0, 1, size=(row, column))
    arr_mask = np.ma.masked_array(arr_mask_ratio, mask=(
        arr_mask_ratio < masked_ratio)).filled(0)
    arr_mask = np.ma.masked_array(
        arr_mask, mask=(arr_mask >= masked_ratio)).filled(1)
    return arr_mask


def calculate_degree_for_virtual_edges(
    edge: np.array,
    label: np.array
):
    """calcualte degree for add same type virtual edges

    Args:
        edge (array): (user,item)
        label (array): (sign) - neg(0), pos(1)

    Returns:
        edgelist_a_b_pos (defaultdict): user -> item positive
        edgelist_a_b_neg (defaultdict): user -> item negative
        edgelist_b_a_pos (defaultdict): item -> user positive
        edgelist_b_a_neg (defaultdict): item -> user negative
        edgelist_a_a_pos (defaultdict): user -> user positive
        edgelist_a_a_neg (defaultdict): user -> user negative
        edgelist_b_b_pos (defaultdict): item -> item positive
        edgelist_b_b_neg (defaultdict): item -> item negative
    """
    edgelist_a_b_pos, edgelist_a_b_neg = defaultdict(list), defaultdict(list)
    edgelist_b_a_pos, edgelist_b_a_neg = defaultdict(list), defaultdict(list)
    edgelist_a_a_pos, edgelist_a_a_neg = defaultdict(list), defaultdict(list)
    edgelist_b_b_pos, edgelist_b_b_neg = defaultdict(list), defaultdict(list)
    
    edge_lists = np.concatenate((edge, label.reshape(-1, 1)), axis=1)
    for a, b, s in edge_lists:
        if s == 1:
            edgelist_a_b_pos[a].append(b)
            edgelist_b_a_pos[b].append(a)
        elif s == 0:
            edgelist_a_b_neg[a].append(b)
            edgelist_b_a_neg[b].append(a)
        else:
            print(a, b, s)
            raise Exception("s must be 0/1")

    edge_list_a_a = defaultdict(lambda: defaultdict(int))

    edge_list_b_b = defaultdict(lambda: defaultdict(int))
    for a, b, s in edge_lists:
        if s == 0:
            s = -1
        for b2 in edgelist_a_b_pos[a]:
            edge_list_b_b[b][b2] += 1 * s
        for b2 in edgelist_a_b_neg[a]:
            edge_list_b_b[b][b2] -= 1 * s
        for a2 in edgelist_b_a_pos[b]:
            edge_list_a_a[a][a2] += 1 * s
        for a2 in edgelist_b_a_neg[b]:
            edge_list_a_a[a][a2] -= 1 * s

    for a1 in edge_list_a_a:
        for a2 in edge_list_a_a[a1]:
            v = edge_list_a_a[a1][a2]
            if a1 == a2:
                continue
            if v > 0:
                edgelist_a_a_pos[a1].append(a2)
            elif v < 0:
                edgelist_a_a_neg[a1].append(a2)

    for b1 in edge_list_b_b:
        for b2 in edge_list_b_b[b1]:
            v = edge_list_b_b[b1][b2]
            if b1 == b2:
                continue
            if v > 0:
                edgelist_b_b_pos[b1].append(b2)
            elif v < 0:
                edgelist_b_b_neg[b1].append(b2)

    return edgelist_a_b_pos, edgelist_a_b_neg, edgelist_b_a_pos, edgelist_b_a_neg, \
        edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg


def dict2array(
    edgelist: defaultdict
):
    edges = []
    for node in edgelist:
        for neighbor in edgelist[node]:
            edges.append([node, neighbor])
    return np.array(edges).T


def perturb_stru_intra(
    edges: tuple,
    augment: str,
    mask_ratio: float
):
    """intra_structre 

    Args:
        edges (tuple): 
        mask_ratio (float): 
        augment (str): 

    Returns:
        _type_: _description_
    """
    edgelist_a_a_pos, edgelist_a_a_neg, \
        edgelist_b_b_pos, edgelist_b_b_neg = edges

    edgelist_a_a_pos = dict2array(edgelist_a_a_pos)
    edgelist_a_a_neg = dict2array(edgelist_a_a_neg)
    edgelist_b_b_pos = dict2array(edgelist_b_b_pos)
    edgelist_b_b_neg = dict2array(edgelist_b_b_neg)

    if augment == 'delete':
        mask_a_pos = generate_mask(
            row=1, column=edgelist_a_a_pos.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_a_neg = generate_mask(
            row=1, column=edgelist_a_a_neg.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_b_pos = generate_mask(
            row=1, column=edgelist_b_b_pos.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_b_neg = generate_mask(
            row=1, column=edgelist_b_b_neg.shape[1], masked_ratio=mask_ratio).squeeze()

        temp_a_pos = edgelist_a_a_pos[:, mask_a_pos != 0]
        temp_a_neg = edgelist_a_a_neg[:, mask_a_neg != 0]
        temp_b_pos = edgelist_b_b_pos[:, mask_b_pos != 0]
        temp_b_neg = edgelist_b_b_neg[:, mask_b_neg != 0]
        return torch.from_numpy(temp_a_pos), torch.from_numpy(temp_a_neg), \
            torch.from_numpy(temp_b_pos), torch.from_numpy(temp_b_neg)

    elif augment == 'flip':
        mask_a_pos = generate_mask(
            row=1, column=edgelist_a_a_pos.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_a_neg = generate_mask(
            row=1, column=edgelist_a_a_neg.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_b_pos = generate_mask(
            row=1, column=edgelist_b_b_pos.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_b_neg = generate_mask(
            row=1, column=edgelist_b_b_neg.shape[1], masked_ratio=mask_ratio).squeeze()

        temp_a_pos = np.concatenate(
            (edgelist_a_a_pos[:, mask_a_pos != 0], edgelist_a_a_neg[:, mask_a_neg == 0]), axis=1)
        temp_a_neg = np.concatenate(
            (edgelist_a_a_pos[:, mask_a_pos == 0], edgelist_a_a_neg[:, mask_a_neg != 0]), axis=1)
        temp_b_pos = np.concatenate(
            (edgelist_b_b_pos[:, mask_b_pos != 0], edgelist_b_b_neg[:, mask_b_neg == 0]), axis=1)
        temp_b_neg = np.concatenate(
            (edgelist_b_b_pos[:, mask_b_pos == 0], edgelist_b_b_neg[:, mask_b_neg != 0]), axis=1)
        return torch.from_numpy(temp_a_pos), torch.from_numpy(temp_a_neg), \
            torch.from_numpy(temp_b_pos), torch.from_numpy(temp_b_neg)

    elif augment == 'add':
        edgelist_a_a_pos = torch.from_numpy(edgelist_a_a_pos)
        edgelist_a_a_neg = torch.from_numpy(edgelist_a_a_neg)
        edgelist_b_b_pos = torch.from_numpy(edgelist_b_b_pos)
        edgelist_b_b_neg = torch.from_numpy(edgelist_b_b_neg)

        pos_add_a = int(mask_ratio * edgelist_a_a_pos.shape[1])
        neg_add_a = int(mask_ratio * edgelist_a_a_neg.shape[1])
        pos_add_b = int(mask_ratio * edgelist_b_b_pos.shape[1])
        neg_add_b = int(mask_ratio * edgelist_b_b_neg.shape[1])

        temp_a = torch.cat([edgelist_a_a_pos, edgelist_a_a_neg], dim=1)
        temp_b = torch.cat([edgelist_b_b_pos, edgelist_b_b_neg], dim=1)

        edges_pos_add_a = negative_sampling(temp_a, num_neg_samples=pos_add_a)
        temp_a = torch.cat([edges_pos_add_a, temp_a], dim=1)
        edges_neg_add_a = negative_sampling(temp_a, num_neg_samples=neg_add_a)

        edges_pos_add_b = negative_sampling(temp_b, num_neg_samples=pos_add_b)
        temp_b = torch.cat([edges_pos_add_b, temp_b], dim=1)
        edges_neg_add_b = negative_sampling(temp_b, num_neg_samples=neg_add_b)

        edgelist_a_a_pos = torch.cat(
            [edgelist_a_a_pos, edges_pos_add_a], dim=1)
        edgelist_a_a_neg = torch.cat(
            [edgelist_a_a_neg, edges_neg_add_a], dim=1)
        edgelist_b_b_pos = torch.cat(
            [edgelist_b_b_pos, edges_pos_add_b], dim=1)
        edgelist_b_b_neg = torch.cat(
            [edgelist_b_b_neg, edges_neg_add_b], dim=1)
        return edgelist_a_a_pos, edgelist_a_a_neg, edgelist_b_b_pos, edgelist_b_b_neg
    else:
        raise "incorrect options"


def perturb_stru_inter(
    edges: tuple,
    augment: str,
    mask_ratio: float
):
    """inter view augmentation

    Args:
        edges (tuple): (edges)
        augment (str): delete, flip, add
        mask_ratio (float): edge mask ration
    Returns:
        augmented graph
    """
    edgelist_a_b_pos, edgelist_a_b_neg = edges
    edgelist_a_b_pos = dict2array(edgelist_a_b_pos)
    edgelist_a_b_neg = dict2array(edgelist_a_b_neg)

    if augment == 'delete':
        mask_pos = generate_mask(
            row=1, column=edgelist_a_b_pos.shape[1], masked_ratio=mask_ratio).squeeze(),
        mask_neg = generate_mask(
            row=1, column=edgelist_a_b_neg.shape[1], masked_ratio=mask_ratio).squeeze()
        return torch.from_numpy(edgelist_a_b_pos[:, mask_pos != 0]), torch.from_numpy(edgelist_a_b_neg[:, mask_neg != 0])

    elif augment == 'flip':
        mask_pos = generate_mask(
            row=1, column=edgelist_a_b_pos.shape[1], masked_ratio=mask_ratio).squeeze()
        mask_neg = generate_mask(
            row=1, column=edgelist_a_b_neg.shape[1], masked_ratio=mask_ratio).squeeze()

        temp_pos = np.concatenate(
            (edgelist_a_b_pos[:, mask_pos != 0], edgelist_a_b_neg[:, mask_neg == 0]), axis=1)
        temp_neg = np.concatenate(
            (edgelist_a_b_pos[:, mask_pos == 0], edgelist_a_b_neg[:, mask_neg != 0]), axis=1)
        return torch.from_numpy(temp_pos), torch.from_numpy(temp_neg)

    elif augment == 'add':
        edgelist_a_b_pos = torch.from_numpy(edgelist_a_b_pos)
        edgelist_a_b_neg = torch.from_numpy(edgelist_a_b_neg)

        temp = torch.cat([edgelist_a_b_pos, edgelist_a_b_neg], dim=1)

        pos_add_a_b = int(mask_ratio * edgelist_a_b_pos.shape[1])
        neg_add_a_b = int(mask_ratio * edgelist_a_b_neg.shape[1])

        edges_pos_add_a_b = negative_sampling(
            temp, num_neg_samples=pos_add_a_b)

        temp = torch.cat([edgelist_a_b_pos, edgelist_a_b_neg, temp], dim=1)
        edges_neg_add_a_b = negative_sampling(
            temp, num_neg_samples=neg_add_a_b)

        edgelist_a_b_pos = torch.cat(
            [edgelist_a_b_pos, edges_pos_add_a_b], dim=1)
        edgelist_a_b_neg = torch.cat(
            [edgelist_a_b_neg, edges_neg_add_a_b], dim=1)
        return edgelist_a_b_pos, edgelist_a_b_neg

    else:
        raise "incorrect option"


class TrnData_lightgcl(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]
    

def norm_matrix(adj_matrix : np.array):
    rowD = np.array(adj_matrix.sum(1)).squeeze()
    colD = np.array(adj_matrix.sum(0)).squeeze()
    for i in range(len(adj_matrix.data)):
        adj_matrix.data[i] = adj_matrix.data[i] / pow(rowD[adj_matrix.row[i]]*colD[adj_matrix.col[i]], 0.5)
    return adj_matrix

def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)