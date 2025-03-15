from keras.datasets import fashion_mnist #question 1
import matplotlib.pyplot as plt
import wandb

def load_data(): # Using fashion mnist to load x train, y train
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0 #784 done to 28 *28, redesigning to 2d format
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0

    # Split data (90% train, 10% validation) recommended, so I done that
    val_size = int(0.1 * x_train.shape[0])
    return (
        x_train[val_size:], y_train[val_size:],  # Train
        x_train[:val_size], y_train[:val_size],  # Validation
        x_test, y_test                           # Test
    )

def Q1_data_classify(Xval, Yval): # training samples will be called
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    fig, axs = plt.subplots(2, 5, figsize=(15, 6)) # I gave size
    for i in range(10):
        img = Xval[Yval == i][2].reshape(28, 28) # 2(3rd) image of class i
        axs[i//5, i%5].imshow(img, cmap='gray')
        axs[i//5, i%5].set_title(class_names[i])
        axs[i//5, i%5].axis('off')
    wandb.log({"Sample Images": wandb.Image(fig)})
    plt.close()