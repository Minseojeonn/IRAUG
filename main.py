#import
import mlflow
import torch 
import dotmap
import numpy as np

#from
from fire import Fire
from parser import parsing
from utils import set_random_seed, collate_fn, select_top_k, precision_recall
from dataset.DataTemplate import DataTemplate
from torch.utils.data import DataLoader 
from model.LightGCN import LightGCN



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
    device = args_enviroments.device

    # Step 1. Preprocessing the dataset and load the dataset
    datatemplate = DataTemplate(args_enviroments.dataset_name, args_enviroments.seed, args_enviroments.split_ratio, args_enviroments.dataset_shuffle, args_enviroments.device, args_enviroments.direction, args_enviroments.input_dim)
    train_dataset, valid_dataset, test_dataset, num_nodes = datatemplate.get_dataset()
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0, collate_fn=collate_fn)
    seen_items = train_dataset.get_seen_nodes()
    
    
    # Step 2. Model definition
    model = LightGCN(args_enviroments, datatemplate).to(device)

    # Step 3. Optimzer definition
    opt = torch.optim.Adam(model.parameters(), lr=args_enviroments.lr)
    
    # Step 4. Training
    for epoch in range(args_enviroments.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            #label is not used, cause it is unsigned model
            opt.zero_grad()
            user, pos, neg = batch
            user, pos, neg = user.to(device), pos.to(device), neg.to(device)
            loss_1, loss_2 = model.bpr_loss(user, pos, neg)
            loss = loss_1 + args_enviroments.wdc * loss_2
            loss.backward()
            opt.step()
            total_loss += loss.item()
        model.eval()
        print(f"Epoch {epoch} Loss: {total_loss / len(train_loader)}")
        if epoch % 1 == 0:
            with torch.no_grad():
                precision, recall = [], []
                for batch in valid_loader:
                    user, items = batch
                    user, items = user.to(device), items
                    pred_rating = model.getUsersRating(user)
                    pred_items = select_top_k(user, pred_rating, args_enviroments.topk, seen_items, num_nodes[0])
                    batch_precision, batch_recall = precision_recall(items, pred_items, num_nodes[0])
                    precision.append(batch_precision)
                    recall.append(batch_recall)
                print(f"Epoch {epoch} Valid Precision: {np.mean(precision)} Recall: {np.mean(recall)}")
        
    # Step 5. Evaluation
    

if __name__ == "__main__":
    Fire(main)