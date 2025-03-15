from keras.datasets import fashion_mnist #question 1
import matplotlib.pyplot as plt
import wandb

wandb.init(project="Deep_Learning_Assignment1_cs24m022", name="Q1_Images")

# Loaded fashion mnist data
(X_train, Y_train), (_, _) = fashion_mnist.load_data()

# All ten classes, I am image classifying it
class_names = [
    "T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]


fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 classes
for i in range(10):
    class_index = np.where(Y_train == i)[0][0]
    image = X_train[class_index]

    # Plot the image in grayscale
    ax = axes[i // 5, i % 5]
    ax.imshow(image, cmap="gray")
    ax.set_title(class_names[i])
    ax.axis("off")

# storing all images to Wandb
wandb.log({"Classify 10 images": wandb.Image(fig)})
plt.close()
wandb.finish()