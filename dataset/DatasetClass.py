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
    def __init__(self, user_item_dict: dict, num_nodes: tuple, device='cpu') -> None:
        self.device = device
        self.user_item_dict = user_item_dict
        self.num_nodes = num_nodes
        self.len = len(user_item_dict)
        # Negative sampling optimized to run during initialization
        self.unseen_items = self.negative_sampling(user_item_dict, num_nodes)
        self.user_list, self.pos, self.neg = self.flatten(user_item_dict, self.unseen_items)
        self.len = len(self.user_list)
            
    def negative_sampling(self, user_item_dict: dict, num_nodes: tuple):
        """
        Optimized negative sampling using numpy for set operations.
        """
        num_users, num_items = num_nodes
        all_items = np.arange(num_users, num_users + num_items)  # 아이템 ID 범위 생성
        
        unseen_item_dict = {}
        for user in user_item_dict:
            seen_items = np.array(user_item_dict[user])  # 사용자가 본 아이템
            unseen_items = np.setdiff1d(all_items, seen_items)  # 보지 않은 아이템 찾기
            unseen_items = np.random.choice(unseen_items, size=len(user_item_dict[user]))  # 보지 않은 아이템 중 100개 샘플링
            unseen_item_dict[user] = unseen_items

        return unseen_item_dict
    
    def flatten(self, user_item_dict, unseen_item_dict):
        """
        Flatten user-item interactions and negative samples for DataLoader.
        """
        user_list = []
        pos_list = []
        neg_list = []
        for user in user_item_dict:
            for pos, neg in zip(user_item_dict[user], unseen_item_dict[user]):
                user_list.append(user)
                pos_list.append(pos)
                neg_list.append(neg)
            
        return user_list, pos_list, neg_list
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # 각 데이터셋의 배치를 반환
        return  self.user_list[index], self.pos[index], self.neg[index]
