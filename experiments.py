import os
from fire import Fire


def main(
    seed
):

    alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   
    for a in alpha:
        os.system(f"python main.py --aug {a} --seed {seed}")

if __name__ == "__main__":
    Fire(main)