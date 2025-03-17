import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.datasets import fashion_mnist, mnist
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import wandb

# All Function we use, activations
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    clip_x = np.clip(x, -50, 50)  # After custom runs, i limited
    return 1 / (1 + np.exp(-clip_x))

def tanh(x):
    clip_x = np.clip(x, -20, 20)
    return np.tanh(clip_x)

def identity(x):
    return x

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)

# -------------------------
# Neural Network Class
# -------------------------
class NeuralNetwork:
    def __init__(self, config):
        """ Best one
        config: dictionary of hyperparameters. For example:
          {
             'inp_size': 784,
             'num_hidden_layers': 3,
             'hidden_size': 64,
             'out_size': 10,
             'batch_sz': 32,
             'lr': 1e-3,
             'weight_init': 'xavier',   # or "random"
             'optimizer': 'adam',        # options: sgd, momentum, nag, rmsprop, adam, nadam
             'epochs': 5,
             'activation': 'relu',       # activation function for hidden layers
             'loss_type': 'cross_entropy',  # or "mean_squared_error"
             'weight_decay': 0.0005,
             'dataset_choice': 'fashion',   # 'fashion' or 'mnist'
             'use_wandb_flag': True,
             'b1': 0.9, 'b2': 0.999, 'beta': 0.9,
             'eps': 1e-8
          }
        """
        self.config = config
        self.inp_size = config['inp_size']
        self.out_size = config['out_size']
        self.num_hidden_layers = config['num_hidden_layers']
        self.hidden_size = config['hidden_size']
        self.batch_sz = config['batch_sz']
        self.lr = config['lr']
        self.weight_init = config['weight_init']
        self.optimizer_choice = config['optimizer']
        self.epochs = config['epochs']
        self.activation = config['activation']
        self.loss_type = config['loss_type']
        self.weight_decay = config['weight_decay']
        self.dataset_choice = config['dataset_choice']
        self.use_wandb_flag = config['use_wandb_flag']
        self.b1 = config.get('b1', 0.9)
        self.b2 = config.get('b2', 0.999)
        self.beta = config.get('beta', 0.9)
        self.eps = config.get('eps', 1e-8)

        self.layer_dims = [self.inp_size] + [self.hidden_size] * self.num_hidden_layers + [self.out_size]
        # Initialize weights and biases for all layers
        self.params = self._initialize_parameters()
        self._prepare_data()

    def _initialize_parameters(self):
        params = {}
        for i in range(1, len(self.layer_dims)):
            fan_in = self.layer_dims[i-1]
            fan_out = self.layer_dims[i]
            if self.weight_init == 'xavier':
                scale = np.sqrt(2 / (fan_in + fan_out))
            else:
                scale = 0.01
            params[i] = {
                'weights': scale * np.random.randn(fan_out, fan_in),
                'biases': np.zeros((fan_out, 1))
            }
        return params

    def _prepare_data(self):
        # Load dataset (Fashion-MNIST or MNIST)
        if self.dataset_choice == 'fashion':
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
        else:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # Normalize and reshape: flatten 28x28 images to 784 size
        X_train = X_train.reshape(-1, 784).T / 255.0
        X_test = X_test.reshape(-1, 784).T / 255.0

        # Use 10% of training data for validation, splitting as said
        X_train, X_val, y_train, y_val = train_test_split(
            X_train.T, y_train, test_size=0.1, stratify=y_train, random_state=42)
        self.X_train = X_train.T
        self.y_train = y_train
        self.X_val = X_val.T
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test

    def forward_pass(self, X):
        activations = [X]
        L = len(self.params)
        for i in range(1, L + 1):
            W = self.params[i]['weights']
            b = self.params[i]['biases']
            z = np.dot(W, activations[-1]) + b
            if i < L:
                if self.activation == 'relu':
                    a = relu(z)
                elif self.activation == 'sigmoid':
                    a = sigmoid(z)
                elif self.activation == 'tanh':
                    a = tanh(z)
                else:
                    a = identity(z)
            else:
                a = z
            activations.append(a)
        # Softmax to get probabilities
        probs = softmax(activations[-1])
        activations[-1] = probs
        return activations

    def compute_loss_and_accuracy(self, X, y):
        activations = self.forward_pass(X)
        probs = activations[-1]
        predictions = np.argmax(probs, axis=0)
        accuracy = np.mean(predictions == y) * 100
        m = y.shape[0]
        if self.loss_type == 'cross_entropy':
            loss = -np.mean(np.log(probs[y, range(m)] + 1e-9))
        else:
            y_onehot = np.eye(self.out_size)[y].T
            loss = np.mean((probs - y_onehot) ** 2)
        return accuracy, loss

    def backward_pass(self, activations, y):
        gradients = {}
        m = y.shape[0]
        L = len(self.params)
        y_onehot = np.eye(self.out_size)[y].T
        delta = activations[-1] - y_onehot  # softmax + cross entropy derivative

        for i in reversed(range(1, L + 1)):
            a_prev = activations[i-1]
            gradients[i] = {
                'dW': np.dot(delta, a_prev.T) / m + self.weight_decay * self.params[i]['weights'],
                'db': np.sum(delta, axis=1, keepdims=True) / m
            }
            if i > 1:
                W = self.params[i]['weights']
                delta = np.dot(W.T, delta)
                if self.activation == 'relu':
                    delta = delta * (a_prev > 0)
                elif self.activation == 'sigmoid':
                    delta = delta * (a_prev * (1 - a_prev))
                elif self.activation == 'tanh':
                    delta = delta * (1 - a_prev**2)
        return gradients

    def update_parameters_sgd(self, gradients):
        for i in self.params:
            self.params[i]['weights'] -= self.lr * gradients[i]['dW']
            self.params[i]['biases']  -= self.lr * gradients[i]['db']

    def update_parameters_momentum(self, gradients, v, beta):
        for i in self.params:
            v[i]['dW'] = beta * v[i]['dW'] + (1 - beta) * gradients[i]['dW']
            v[i]['db'] = beta * v[i]['db'] + (1 - beta) * gradients[i]['db']
            self.params[i]['weights'] -= self.lr * v[i]['dW']
            self.params[i]['biases']  -= self.lr * v[i]['db']
        return v

    def update_parameters_rmsprop(self, gradients, s, beta):
        for i in self.params:
            s[i]['dW'] = beta * s[i]['dW'] + (1 - beta) * (gradients[i]['dW'] ** 2)
            s[i]['db'] = beta * s[i]['db'] + (1 - beta) * (gradients[i]['db'] ** 2)
            self.params[i]['weights'] -= self.lr * gradients[i]['dW'] / (np.sqrt(s[i]['dW']) + self.eps)
            self.params[i]['biases']  -= self.lr * gradients[i]['db'] / (np.sqrt(s[i]['db']) + self.eps)
        return s

    def update_parameters_adam(self, gradients, v, s, t):
        for i in self.params:
            v[i]['dW'] = self.b1 * v[i]['dW'] + (1 - self.b1) * gradients[i]['dW']
            v[i]['db'] = self.b1 * v[i]['db'] + (1 - self.b1) * gradients[i]['db']
            s[i]['dW'] = self.b2 * s[i]['dW'] + (1 - self.b2) * (gradients[i]['dW'] ** 2)
            s[i]['db'] = self.b2 * s[i]['db'] + (1 - self.b2) * (gradients[i]['db'] ** 2)
            v_corr_W = v[i]['dW'] / (1 - self.b1 ** t)
            v_corr_b = v[i]['db'] / (1 - self.b1 ** t)
            s_corr_W = s[i]['dW'] / (1 - self.b2 ** t)
            s_corr_b = s[i]['db'] / (1 - self.b2 ** t)
            self.params[i]['weights'] -= self.lr * v_corr_W / (np.sqrt(s_corr_W) + self.eps)
            self.params[i]['biases']  -= self.lr * v_corr_b / (np.sqrt(s_corr_b) + self.eps)
        return v, s
    def update_parameters_nesterov(self, gradients, v, beta):
        for i in self.params:
            # Update momentum using the current gradient
            v[i]['dW'] = beta * v[i]['dW'] + (1 - beta) * gradients[i]['dW']
            v[i]['db'] = beta * v[i]['db'] + (1 - beta) * gradients[i]['db']
            # Nesterov update: incorporate the lookahead momentum term along with the gradient
            self.params[i]['weights'] -= self.lr * (beta * v[i]['dW'] + gradients[i]['dW'])
            self.params[i]['biases']  -= self.lr * (beta * v[i]['db'] + gradients[i]['db'])
        return v

    def update_parameters_nadam(self, gradients, v, s, t):
        for i in self.params:
            # Update first moment estimate
            v[i]['dW'] = self.b1 * v[i]['dW'] + (1 - self.b1) * gradients[i]['dW']
            v[i]['db'] = self.b1 * v[i]['db'] + (1 - self.b1) * gradients[i]['db']
            # Bias-corrected first moment estimates
            m_hat_W = v[i]['dW'] / (1 - self.b1 ** t)
            m_hat_b = v[i]['db'] / (1 - self.b1 ** t)

            # Update second moment estimate
            s[i]['dW'] = self.b2 * s[i]['dW'] + (1 - self.b2) * (gradients[i]['dW'] ** 2)
            s[i]['db'] = self.b2 * s[i]['db'] + (1 - self.b2) * (gradients[i]['db'] ** 2)
            s_hat_W = s[i]['dW'] / (1 - self.b2 ** t)
            s_hat_b = s[i]['db'] / (1 - self.b2 ** t)

            # Nadam have that lookahead correction:
            m_bar_W = (self.b1 * m_hat_W + (1 - self.b1) * gradients[i]['dW'] / (1 - self.b1 ** t))
            m_bar_b = (self.b1 * m_hat_b + (1 - self.b1) * gradients[i]['db'] / (1 - self.b1 ** t))

            # Update parameters using the Nadam rule
            self.params[i]['weights'] -= self.lr * m_bar_W / (np.sqrt(s_hat_W) + self.eps)
            self.params[i]['biases']  -= self.lr * m_bar_b / (np.sqrt(s_hat_b) + self.eps)
        return v, s


    def fit(self):
        if self.optimizer_choice == 'sgd':
            self._train_sgd()
        elif self.optimizer_choice == 'momentum':
            self._train_momentum()
        elif self.optimizer_choice == 'rmsprop':
            self._train_rmsprop()
        elif self.optimizer_choice == 'adam':
            self._train_adam()
        elif self.optimizer_choice == 'nesterov':
            self._train_nesterov()
        elif self.optimizer_choice == 'nadam':
            self._train_nadam()

    def _train_sgd(self):
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                self.update_parameters_sgd(grads)
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                           "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")

    def _train_momentum(self):
        v = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                 'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                v = self.update_parameters_momentum(grads, v, self.beta)
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                           "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")

    def _train_rmsprop(self):
        s = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                 'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                s = self.update_parameters_rmsprop(grads, s, self.beta)
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                           "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")

    def _train_adam(self):
        v = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                 'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        s = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                 'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        t = 1
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                v, s = self.update_parameters_adam(grads, v, s, t)
                t += 1
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                           "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")

    def _train_nesterov(self):
        # first of all, give momentum dvalue as zero
        v = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                v = self.update_parameters_nesterov(grads, v, self.beta)
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                          "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")

    def _train_nadam(self):
        # Initialize first and second moment dictionaries with zeros
        v = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        s = {i: {'dW': np.zeros_like(self.params[i]['weights']),
                'db': np.zeros_like(self.params[i]['biases'])} for i in self.params}
        t = 1
        for epoch in range(self.epochs):
            permutation = np.random.permutation(self.X_train.shape[1])
            X_shuffled = self.X_train[:, permutation]
            y_shuffled = self.y_train[permutation]
            for i in range(0, self.X_train.shape[1], self.batch_sz):
                X_batch = X_shuffled[:, i:i+self.batch_sz]
                y_batch = y_shuffled[i:i+self.batch_sz]
                activations = self.forward_pass(X_batch)
                grads = self.backward_pass(activations, y_batch)
                v, s = self.update_parameters_nadam(grads, v, s, t)
                t += 1
            train_acc, train_loss = self.compute_loss_and_accuracy(self.X_train, self.y_train)
            val_acc, val_loss = self.compute_loss_and_accuracy(self.X_val, self.y_val)
            if self.use_wandb_flag:
                wandb.log({"epoch": epoch, "loss": train_loss, "accuracy": train_acc,
                          "val_loss": val_loss, "val_accuracy": val_acc})
            print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}%, Loss: {val_loss:.4f}")


    def predict(self, X):
        activations = self.forward_pass(X)
        probs = activations[-1]
        return np.argmax(probs, axis=0)

    def log_confusion_matrix(self):
        preds = self.predict(self.X_test)
        if self.dataset_choice == 'fashion':
            class_names = ["T-shirt/Top", "Trouser", "Pullover", "Dress", "Coat",
                           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]
        else:
            class_names = [str(i) for i in range(10)]
        cm = confusion_matrix(self.y_test, preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        filename = "confusion_matrix.png"
        plt.savefig(filename)
        plt.close()
        if self.use_wandb_flag:
            wandb.log({"Confusion Matrix": wandb.Image(filename)})
        else:
            plt.show()
