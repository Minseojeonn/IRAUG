import random
import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def set_random_seed(seed, device):
    if device == 'cpu':
        pass
    else:
        device = device.split(':')[0]

    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def logging_with_mlflow(epochs, val_metric, test_metric):
    pass

def collate_fn(batch):
    user, items = zip(*batch)
    user = torch.LongTensor(user)
    return user, items

def precision_recall(actual, recommended):
    """ 추천 시스템에서 Precision과 Recall을 계산하는 함수
    actual: 실제로 선호한 항목 (리스트)
    recommended: 추천된 항목 (리스트)
    k: 추천 항목의 개수 제한
    """
    recommended_at_k = recommended  # 상위 k개의 추천 항목만 고려
    
    # 교집합: 추천된 항목 중에서 실제 선호한 항목
    relevant_and_recommended = set(recommended_at_k).intersection(set(actual))
    
    # Precision: 추천된 항목 중에서 실제로 선호한 항목의 비율
    precision = len(relevant_and_recommended) / len(recommended_at_k) if recommended_at_k else 0
    
    # Recall: 실제로 선호한 항목 중에서 추천된 항목의 비율
    recall = len(relevant_and_recommended) / len(actual) if actual else 0
    
    return precision, recall

def select_top_k(pred, top_k):
    """
    Select top k items from the predicted items.
    args:
        pred (torch.tensor): (num_user, all_items)
        top_k (int): top k items
    """
    top_k_val, top_k_idx = torch.topk(pred, top_k, dim=1)
    return top_k_idx