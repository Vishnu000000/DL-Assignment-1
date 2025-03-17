# main.py
import argparse
from network import NeuralNetwork
import sweep
import visualize

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Assignment 1")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "sweep", "visualize", "loss_comparison"],
                        help="Mode to run: 'train' to run a single training run, 'sweep' to run the sweep agent, 'visualize' to visualize artifacts, or 'loss_comparison' for loss function analysis.")
    args = parser.parse_args()
    
    if args.mode == "train":
        # Example training run with a fixed configuration.
        config = {
            'inp_size': 784,
            'num_hidden_layers': 3,
            'hidden_size': 64,
            'out_size': 10,
            'batch_sz': 32,
            'lr': 1e-3,
            'weight_init': 'xavier',
            'optimizer': 'adam',
            'epochs': 5,
            'activation': 'relu',
            'loss_type': 'cross_entropy',
            'weight_decay': 0.0005,
            'dataset_choice': 'fashion',
            'use_wandb_flag': True,
            'b1': 0.9,
            'b2': 0.999,
            'beta': 0.9,
            'eps': 1e-8
        }
        model = NeuralNetwork(config)
        model.fit()
        model.log_confusion_matrix()
    elif args.mode == "sweep":
        sweep.run_sweep()
    elif args.mode == "visualize":
        visualize.visualize_confusion_matrix()
    elif args.mode == "loss_comparison":
        # If you have a loss comparison function in visualize.py, call it here.
        # For example, assume visualize.compare_loss_functions() exists.
        try:
            visualize.compare_loss_functions()  # You can implement this function in visualize.py
        except AttributeError:
            print("Loss comparison function is not implemented in visualize.py")

if __name__ == "__main__":
    main()
