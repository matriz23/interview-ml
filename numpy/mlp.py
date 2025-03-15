import numpy as np
from tensorflow.keras.dAtasets import mnist


def softmax(X):
    exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def relu(X):
    return np.maximum(0, X)


def cross_entropy_loss(y_pred, y):
    return -np.mean(y * np.log(y_pred + 1e-8))


class MLP:
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # He
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(
            2.0 / input_size
        )  # (input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))  # (1, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(
            2.0 / hidden_size
        )  # (hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))  # (1, output_size)

        self.cache = {}

    def forward(self, X):  # (bs, input_size)
        Z1 = np.dot(X, self.W1) + self.b1  # (bs, hidden_size)
        A1 = relu(Z1)  # (bs, hidden_size)
        Z2 = np.dot(A1, self.W2) + self.b2  # (bs, output_size)
        A2 = softmax(Z2)  # (bs, output_size)

        self.cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2

    def backward(self, X, y, lr=0.1):
        m = y.shape[0]
        A1 = self.cache["A1"]  # (bs, hidden_size)
        A2 = self.cache["A2"]  # (bs, output_size)
        Z1 = self.cache["Z1"]  # (bs, hidden_size)

        dZ2 = A2 - y  # # (bs, output_size)
        dW2 = np.dot(A1.T, dZ2) / m  # (hidden_size, output_size)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m  # (1, output_size)

        dA1 = np.dot(dZ2, self.W2.T)  # (bs, hidden_size)
        dZ1 = dA1 * (Z1 > 0)  # (bs, hidden_size)
        dW1 = np.dot(X.T, dZ1) / m  # (input_size, hidden_size)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m  # (1, hidden_size)

        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        A2 = self.forward(X)
        return np.argmax(A2, axis=1)

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
                loss = cross_entropy_loss(self.cache["A2"], y_batch)

                total_loss += loss
                self.backward(x_batch, y_batch, lr)

            train_acc = self.evaluate(X_train, np.argmax(y_train, axis=1))
            msg = f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / num_batches:.4f}, Train Acc: {train_acc:.4f}"

            print(msg)

    def evaluate(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_dAta()
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0
    y_train_onehot = np.eye(10)[y_train]

    mlp = MLP()
    mlp.fit(X_train=X_train, y_train=y_train_onehot, epochs=20, batch_size=64, lr=0.1)

    test_acc = mlp.evaluate(X_test, y_test)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
