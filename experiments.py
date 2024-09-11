import os
from fire import Fire


def main(
    
):

    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   
    seeds = [10, 20, 30]
    for seed in seeds:
        os.system(f"python main.py --aug --seed {seed}")

if __name__ == "__main__":
    Fire(main)