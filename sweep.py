# sweep.py
import wandb
from network import NeuralNetwork  # Import the NeuralNetwork class from network.py

# Sweep configuration dictionary.
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "val_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "epochs": {"values": [5, 10]},
        "num_hidden_layers": {"values": [3, 4, 5]},
        "hidden_size": {"values": [32, 64, 128]},
        "weight_decay": {"values": [0, 0.0005, 0.5]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]},
        "batch_size": {"values": [16, 32, 64]},
        "weight_init": {"values": ["random", "xavier"]},
        "activation": {"values": ["sigmoid", "tanh", "relu"]},
        "loss_type": {"values": ["cross_entropy", "mean_squared_error"]}
    }
}

def run_sweep():
    # Initialize sweep and get sweep ID.
    sweep_id = wandb.sweep(sweep_config, project="Deep_Learning_Assignment1_cs24m022")
    print("Sweep ID:", sweep_id)
    
    def sweep_train():
        run = wandb.init(project="Deep_Learning_Assignment1_cs24m022")
        config = run.config
        run_name = (
            f"hl_{config.num_hidden_layers}_"
            f"bs_{config.batch_size}_"
            f"hs_{config.hidden_size}_"
            f"loss_{config.loss_type}_"
            f"init_{config.weight_init}_"
            f"ac_{config.activation}_"
            f"wd_{config.weight_decay}_"
            f"lr_{config.learning_rate}_"
            f"opt_{config.optimizer}"
        )
        wandb.run.name = run_name
        
        model_config = {
            'inp_size': 784,
            'num_hidden_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'out_size': 10,
            'batch_sz': config.batch_size,
            'lr': config.learning_rate,
            'weight_init': config.weight_init,
            'optimizer': config.optimizer,
            'epochs': config.epochs,
            'activation': config.activation,
            'loss_type': config.loss_type,
            'weight_decay': config.weight_decay,
            'dataset_choice': 'fashion',
            'use_wandb_flag': True,
            'b1': 0.9,
            'b2': 0.999,
            'beta': 0.9,
            'eps': 1e-8
        }
        model = NeuralNetwork(model_config)
        model.fit()
        
        # Log final validation metrics.
        val_acc, val_loss = model.compute_loss_and_accuracy(model.X_val, model.y_val)
        wandb.log({"final_val_accuracy": val_acc, "final_val_loss": val_loss})
        
        # Log the confusion matrix.
        model.log_confusion_matrix()
        artifact = wandb.Artifact("confusion_matrix", type="image")
        artifact.add_file("confusion_matrix.png")
        artifact.metadata = {"val_accuracy": val_acc}
        wandb.log_artifact(artifact)
        
        wandb.finish()
    
    wandb.agent(sweep_id, function=sweep_train, count=100)

if __name__ == "__main__":
    run_sweep()
