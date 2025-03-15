import numpy as np


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def cross_entropy_loss(y_pred, y):
    return -np.mean((y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8)))


class LogisticRegression:
    def __init__(self, input_size):
        self.W = np.random.randn(input_size, 1)
        self.b = np.zeros((1, 1))
        self.cache = {}

    def forward(self, X):
        z = np.dot(self.W, X) + self.b
        a = sigmoid(z)
        self.cache = {"z": z, "a": a}
        return a

    def backward(self, X, y, lr):
        # Loss = -(yloga + (1-y)log(1-a))
        # dL/da = -(y/a - (1-y)/(1-a))
        a = self.cache["a"]
        z = self.cache["z"]
        dz = a - y  # (bs, 1)
        dW = np.dot(X.T, dz)  # (input_size, 1)
        db = np.sum(dz, axis=0, keepdims=True)
        self.W -= dW * lr
        self.b -= db * lr

    def predict(self, X):
        a = self.forward(X)
        return np.where(a > 0.5, 1.0, 0.0)

    def fit(self, X_train, y_train, epochs=20, batch_size=64, lr=0.1):
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_train_shuffled = X_train[permutation]
            y_train_shuffled = y_train[permutation]

            total_loss = 0

            for batch in range(num_batches):
                x_batch = X_train_shuffled[
                    batch * batch_size : (batch + 1) * batch_size
                ]
                y_batch = y_train_shuffled[
                    batch * batch_size : (batch + 1) * batch_size
                ]

                self.forward(x_batch)
                loss = cross_entropy_loss(self.cache["a"], y_batch)

                total_loss += loss
                self.backward(x_batch, y_batch, lr)

            train_acc = self.evaluate(X_train, y_train)
            msg = f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}, Train Acc: {train_acc:.4f}"

            print(msg)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)
