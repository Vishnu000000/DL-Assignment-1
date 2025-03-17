# q1_images.py
import wandb
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

def log_sample_images():
    # Initialize a WandB run for logging.
    wandb.init(project="Deep_Learning_Assignment1_cs24m022", name="Q1_Images")
    
    # Load Fashion-MNIST dataset.
    (X_train, y_train), (_, _) = fashion_mnist.load_data()
    
    # Define class names for 10 classes.
    class_names = [
        "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
    ]
    
    # Create a figure with 2 rows x 5 columns.
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # For each class (0-9), select the first occurrence and plot it.
    for i in range(10):
        idx = (y_train == i).nonzero()[0][0]
        image = X_train[idx]
        ax = axes[i // 5, i % 5]
        ax.imshow(image, cmap="gray")
        ax.set_title(class_names[i], fontsize=12, fontweight="bold")
        ax.axis("off")
    
    plt.suptitle("One Sample Image per Class (Fashion-MNIST)", fontsize=16, fontweight="bold")
    
    # Log the figure to WandB.
    wandb.log({"Classify 10 Images": wandb.Image(fig)})
    plt.close(fig)
    wandb.finish()

if __name__ == "__main__":
    log_sample_images()
