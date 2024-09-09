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

# ====================Metrics==============================
# =========================================================
def compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    return precision, recall


def collate_fn(batch):
    user, items = zip(*batch)
    user = torch.LongTensor(user)
    return user, items