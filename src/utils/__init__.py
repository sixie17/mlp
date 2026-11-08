"""
From-scratch multilayer perceptron library (NumPy only).

Modules:
    activations    - activation functions and their derivatives
    initializers   - weight initialization schemes
    layers         - DenseLayer (forward + backward pass)
    losses         - cross-entropy losses
    optimizers     - SGD (mandatory) + Momentum / Nesterov / RMSprop / Adam (bonus)
    network        - Network (feedforward / backpropagation) + save / load
    training       - fit loop, evaluation, early stopping
    metrics        - accuracy, precision, recall, f1, ROC AUC
    preprocessing  - dataset loading, splitting, scaling, one-hot encoding
    plots          - learning curves and model comparison graphs
    pipeline       - high-level split / train / predict / compare commands
"""
