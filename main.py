#import
import mlflow
import torch 
import dotmap

#from
from fire import Fire
from parser import parsing
from utils import set_random_seed
from dataset.DataTemplate import DataTemplate
from torch.utils.data import DataLoader 


def main():
    args_enviroments = dotmap.DotMap(vars(parsing()))
    #set env parameters
   
    # Set MLflow
    if args_enviroments.use_mlflow:
        remote_server_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(remote_server_uri)
        experiment_name = f"exper_name"
        mlflow.set_experiment(experiment_name)
        mlflow.start_run()

    # Step 0. Initialization
    args_enviroments.device = args_enviroments.device if torch.cuda.is_available() else 'cpu'
    set_random_seed(seed=args_enviroments.seed, device=args_enviroments.device)

    # Step 1. Preprocessing the dataset and load the dataset
    datatemplate = DataTemplate(args_enviroments.dataset_name, args_enviroments.seed, args_enviroments.split_ratio, args_enviroments.dataset_shuffle, args_enviroments.device, args_enviroments.direction, args_enviroments.node_idx_type, args_enviroments.input_dim)
    train_dataset, valid_dataset, test_dataset = datatemplate.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    breakpoint()
    
    # Step 2. Model definition
    



if __name__ == "__main__":
    Fire(main)