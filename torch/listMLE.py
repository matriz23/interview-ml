import torch

def listMLE_loss(y_true, y_pred):
    """
    Computes the ListMLE loss for a given set of true and predicted values.

    ListMLE is a listwise learning-to-rank loss function that optimizes the 
    likelihood of the correct ranking of items.

    Args:
        y_true (torch.Tensor): A tensor of true relevance scores with shape (batch_size, list_size).
        y_pred (torch.Tensor): A tensor of predicted relevance scores with shape (batch_size, list_size).

    Returns:
        torch.Tensor: A scalar tensor representing the mean ListMLE loss over the batch.

    Example:
        y_true = torch.tensor([[3, 2, 1], [1, 3, 2]])
        y_pred = torch.tensor([[0.1, 0.4, 0.2], [0.3, 0.2, 0.5]])
        loss = listMLE_loss(y_true, y_pred)
    """
    _, indices = torch.sort(y_true, dim=-1, descending=True, stable=True)

    sorted_y_pred = torch.gather(y_pred, dim=-1, index=indices)

    rev_sorted_y_pred = torch.flip(sorted_y_pred, dims=[-1])

    rev_log_cum_sum_exp = torch.logcumsumexp(rev_sorted_y_pred, dim=-1)

    log_cum_sum_exp = torch.flip(rev_log_cum_sum_exp, dims=[-1])
    loss = (-sorted_y_pred + log_cum_sum_exp).sum(dim=-1)
    return loss.mean()

y_true = torch.tensor([[3.0, 2.0, 4.0]])
y_pred = torch.tensor([[0.5, 0.3, 1.2]])
print(listMLE_loss(y_true, y_pred))  # 输出约为1.241
    