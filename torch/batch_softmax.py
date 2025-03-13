import torch
import torch.nn.functional as F


def batch_softmax(user_embeddings, item_embeddings, temperature):
    logit = torch.matmul(user_embeddings, item_embeddings.transpose())
    # bs, d 
    # bs, bs 
    # 对角线 label 为 1，其他为 0
    bs, d = user_embeddings.shape
    onehot_labels = torch.eye(bs)
    logit_with_temperature = logit / temperature
    loss_per_example = F.cross_entropy(logit,)
    