import numpy as np

def auc(y_true, y_pred):
    sorted_idx = np.argsort(y_pred)[::-1]
    y_true_sorted = np.array(y_true)[sorted_idx]
    
    pos_num = np.sum(y_true_sorted)
    neg_num = len(y_true_sorted) - pos_num
    
    # cumsum 计算到 a_i 为止的累积和, 对于本题而言, 等于到 i 为止的正例数
    n_cum = np.cumsum(y_true_sorted) 
    
    auc = np.sum(n_cum[y_true_sorted == 0]) / (neg_num * pos_num)
    
    return auc

if __name__ == '__main__':
    y_true = [0, 0, 1, 1]
    y_pred = [0.1, 0.4, 0.3, 0.8]
    print(auc(y_true=y_true, y_pred=y_pred))
    