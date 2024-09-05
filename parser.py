import argparse

# Parsing arguments
def parsing(): 
    parser = argparse.ArgumentParser(description='Python parser usage.')
    parser.add_argument('--device', default="cuda:0", type=str, help='device')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--use_mlflow', default=False, type=bool, help='use_mlflow')
    parser.add_argument('--dataset_name', default="ml-1m", type=str, help='dataset_name')
    parser.add_argument('--split_ratio', default=[0.8, 0.1, 0.1], type=list, help='split_ratio')
    parser.add_argument('--dataset_shuffle', default=True, type=bool, help='dataset_shuffle')
    parser.add_argument('--direction', default=True, type=bool, help='direction')
    parser.add_argument('--node_idx_type', default="uni", type=str, help='node_idx_type')
    parser.add_argument('--input_dim', default=256, type=int, help='input_dim')
    parser.add_argument('--batch_size', default=256, type=int, help='batch_size')
    args = parser.parse_args()
    return args