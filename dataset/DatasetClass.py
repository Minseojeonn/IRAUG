import torch
import numpy as np
import torch.utils.data as data

class DatasetClass(data.Dataset):
    """
    Dataset class with optimized negative sampling using numpy for set operations.
    
    args:
        edge (np.array): edge
        label (np.array): label
        user_item_dict (dict): dictionary with user-item interactions
        num_nodes (tuple): (num_users, num_items)
    
    return:
        Dataset ready for DataLoader
    """
    def __init__(self, edge: np.array, label: np.array, user_item_dict: dict, num_nodes: tuple, device='cpu') -> None:
        self.edge = torch.tensor(edge, dtype=torch.long, device=device)
        self.label = torch.tensor(label, dtype=torch.float, device=device)
        self.len = edge.shape[0]
        self.device = device
        
        # Negative sampling optimized to run during initialization
        self.unseen_items = self.negative_sampling(user_item_dict, num_nodes)
    
    def negative_sampling(self, user_item_dict: dict, num_nodes: tuple):
        """
        Optimized negative sampling using numpy for set operations.
        """
        num_users, num_items = num_nodes
        all_items = np.arange(num_users, num_users + num_items)  # 아이템 ID 범위 생성
        
        unseen_item_list = []
        for user in user_item_dict:
            seen_items = np.array(user_item_dict[user])  # 사용자가 본 아이템
            unseen_items = np.setdiff1d(all_items, seen_items)  # 보지 않은 아이템 찾기
            user_item_dict[user] = unseen_items  # unseen_items를 미리 저장
            
        # 각 엣지에 대해 보지 않은 아이템에서 하나씩 샘플링
        for ed in self.edge.cpu().numpy():
            user = ed[0]
            unseen_items = user_item_dict[user]
            unseen_item = np.random.choice(unseen_items)  # 랜덤 아이템 선택
            unseen_item_list.append(unseen_item)
        
        assert len(unseen_item_list) == self.len, "unseen item list length must be same with edge length"
        
        return torch.tensor(unseen_item_list, dtype=torch.long, device=self.device)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # 각 데이터셋의 배치를 반환
        return self.edge[index], self.label[index], self.unseen_items[index]
