import numpy as np

def _sigmoid(z):
    return np.where(z >= 0,
                    1/(1+np.exp(-z)),
                    np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    
    # Ensure numpy arrays
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    N, D = X.shape
    
    w = np.zeros((D, 1))
    b = 0.0
    eps = 1e-9

    for _ in range(steps):
        z = X @ w + b
        y_hat = _sigmoid(z)

        loss = -np.mean(
            y * np.log(y_hat + eps) +
            (1 - y) * np.log(1 - y_hat + eps)
        )

        dw = (X.T @ (y_hat - y)) / N
        db = np.mean(y_hat - y)

        w -= lr * dw
        b -= lr * db

    return w.flatten().tolist(), float(b)