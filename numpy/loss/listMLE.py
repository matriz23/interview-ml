import numpy as np


def listMLE_loss(y_true, y_pred):
    """
    Computes the ListMLE loss for a given set of true and predicted values.

    ListMLE is a listwise learning-to-rank loss function that optimizes the
    likelihood of the correct ranking of items.

    Args:
        y_true (np.ndarray): An array of true relevance scores with shape (batch_size, list_size).
        y_pred (np.ndarray): An array of predicted relevance scores with shape (batch_size, list_size).

    Returns:
        float: The mean ListMLE loss over the batch.

    Example:
        y_true = np.array([[3, 2, 1]])
        y_pred = np.array([[0.1, 0.4, 0.2]])
        loss = listMLE_loss(y_true, y_pred)
    """

    indices = np.argsort(-y_true, axis=-1, kind="stable")

    sorted_y_pred = np.take_along_axis(y_pred, indices, axis=-1)

    rev_sorted_y_pred = np.flip(sorted_y_pred, axis=-1)

    rev_log_cum_sum_exp = np.log(np.cumsum(np.exp(rev_sorted_y_pred), axis=-1))

    log_cum_sum_exp = np.flip(rev_log_cum_sum_exp, axis=-1)

    loss = (-sorted_y_pred + log_cum_sum_exp).sum(axis=-1)
    return np.mean(loss)


if __name__ == "__main__":
    y_true = np.array([[3.0, 2.0, 4.0]])
    y_pred = np.array([[0.5, 0.3, 1.2]])
    print(listMLE_loss(y_true, y_pred))
