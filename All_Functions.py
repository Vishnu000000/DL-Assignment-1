import numpy as np

def train_model(model, optimizer, lr, batch_size, epochs, 
                momentum, beta, beta1, beta2, epsilon,
                X_train, y_train, X_val, y_val):
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []
    
    # Initialize optimizer states
    if optimizer in ['momentum', 'nag']:
        v_w = [np.zeros_like(w) for w in model.weights]
        v_b = [np.zeros_like(b) for b in model.biases]
    elif optimizer == 'rmsprop':
        cache_w = [np.zeros_like(w) for w in model.weights]
        cache_b = [np.zeros_like(b) for b in model.biases]
    elif optimizer in ['adam', 'nadam']:
        m_w = [np.zeros_like(w) for w in model.weights]
        m_b = [np.zeros_like(b) for b in model.biases]
        v_w = [np.zeros_like(w) for w in model.weights]
        v_b = [np.zeros_like(b) for b in model.biases]
        t = 0

    for epoch in range(epochs):
        # Shuffle training data
        indices = np.random.permutation(len(X_train))
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_loss = 0
        for i in range(0, len(X_train), batch_size):
            # Get mini-batch
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # Forward + backward pass
            model.forward(X_batch)
            grads_w, grads_b = model.backprop(y_batch)
            
            # Optimizer updates
            if optimizer == 'sgd':
                update_sgd(model, grads_w, grads_b, lr)
                
            elif optimizer == 'momentum':
                v_w, v_b = update_momentum(model, grads_w, grads_b, 
                                         lr, momentum, v_w, v_b)
                
            elif optimizer == 'nag':
                v_w, v_b = update_nag(model, grads_w, grads_b,
                                    lr, momentum, v_w, v_b)
                
            elif optimizer == 'rmsprop':
                cache_w, cache_b = update_rmsprop(model, grads_w, grads_b,
                                                lr, beta, epsilon, 
                                                cache_w, cache_b)
                
            elif optimizer == 'adam':
                t += 1
                m_w, m_b, v_w, v_b = update_adam(model, grads_w, grads_b,
                                                lr, beta1, beta2, epsilon,
                                                m_w, m_b, v_w, v_b, t)
                
            elif optimizer == 'nadam':
                t += 1
                m_w, m_b, v_w, v_b = update_nadam(model, grads_w, grads_b,
                                                lr, beta1, beta2, epsilon,
                                                m_w, m_b, v_w, v_b, t)
            
            epoch_loss += model.compute_loss(y_batch)
        
        # Epoch metrics
        train_acc.append(model.evaluate(X_train, y_train)[0])
        train_loss.append(epoch_loss/(len(X_train)//batch_size))
        val_acc.append(model.evaluate(X_val, y_val)[0])
        val_loss.append(model.compute_loss(y_val))
        
    return train_acc, train_loss, val_acc, val_loss

def update_sgd(model, grads_w, grads_b, lr):
    for i in range(len(model.weights)):
        model.weights[i] -= lr * grads_w[i]
        model.biases[i] -= lr * grads_b[i]

def update_momentum(model, grads_w, grads_b, lr, momentum, v_w, v_b):
    for i in range(len(model.weights)):
        v_w[i] = momentum*v_w[i] + lr*grads_w[i]
        v_b[i] = momentum*v_b[i] + lr*grads_b[i]
        model.weights[i] -= v_w[i]
        model.biases[i] -= v_b[i]
    return v_w, v_b

def update_nag(model, grads_w, grads_b, lr, momentum, v_w, v_b):
    for i in range(len(model.weights)):
        v_w_prev = v_w[i].copy()
        v_b_prev = v_b[i].copy()
        
        v_w[i] = momentum*v_w[i] + lr*grads_w[i]
        v_b[i] = momentum*v_b[i] + lr*grads_b[i]
        
        model.weights[i] -= (1 + momentum)*v_w[i] - momentum*v_w_prev
        model.biases[i] -= (1 + momentum)*v_b[i] - momentum*v_b_prev
    return v_w, v_b

def update_rmsprop(model, grads_w, grads_b, lr, beta, epsilon, cache_w, cache_b):
    for i in range(len(model.weights)):
        cache_w[i] = beta*cache_w[i] + (1-beta)*grads_w[i]**2
        cache_b[i] = beta*cache_b[i] + (1-beta)*grads_b[i]**2
        
        model.weights[i] -= lr * grads_w[i]/(np.sqrt(cache_w[i]) + epsilon)
        model.biases[i] -= lr * grads_b[i]/(np.sqrt(cache_b[i]) + epsilon)
    return cache_w, cache_b

def update_adam(model, grads_w, grads_b, lr, beta1, beta2, epsilon, m_w, m_b, v_w, v_b, t):
    for i in range(len(model.weights)):
        # Update first moment
        m_w[i] = beta1*m_w[i] + (1-beta1)*grads_w[i]
        m_b[i] = beta1*m_b[i] + (1-beta1)*grads_b[i]
        
        # Update second moment
        v_w[i] = beta2*v_w[i] + (1-beta2)*grads_w[i]**2
        v_b[i] = beta2*v_b[i] + (1-beta2)*grads_b[i]**2
        
        # Bias correction
        m_w_hat = m_w[i]/(1 - beta1**t)
        m_b_hat = m_b[i]/(1 - beta1**t)
        v_w_hat = v_w[i]/(1 - beta2**t)
        v_b_hat = v_b[i]/(1 - beta2**t)
        
        # Update parameters
        model.weights[i] -= lr * m_w_hat/(np.sqrt(v_w_hat) + epsilon)
        model.biases[i] -= lr * m_b_hat/(np.sqrt(v_b_hat) + epsilon)
    return m_w, m_b, v_w, v_b

def update_nadam(model, grads_w, grads_b, lr, beta1, beta2, epsilon, m_w, m_b, v_w, v_b, t):
    for i in range(len(model.weights)):
        # Update moments with current gradients
        m_w[i] = beta1*m_w[i] + (1-beta1)*grads_w[i]
        m_b[i] = beta1*m_b[i] + (1-beta1)*grads_b[i]
        v_w[i] = beta2*v_w[i] + (1-beta2)*grads_w[i]**2
        v_b[i] = beta2*v_b[i] + (1-beta2)*grads_b[i]**2
        
        # Compute bias-corrected moments with Nesterov
        m_w_hat = (beta1*m_w[i]/(1 - beta1**(t+1))) + ((1-beta1)*grads_w[i]/(1 - beta1**t))
        m_b_hat = (beta1*m_b[i]/(1 - beta1**(t+1))) + ((1-beta1)*grads_b[i]/(1 - beta1**t))
        v_w_hat = v_w[i]/(1 - beta2**t)
        v_b_hat = v_b[i]/(1 - beta2**t)
        
        # Update parameters
        model.weights[i] -= lr * m_w_hat/(np.sqrt(v_w_hat) + epsilon)
        model.biases[i] -= lr * m_b_hat/(np.sqrt(v_b_hat) + epsilon)
    return m_w, m_b, v_w, v_b