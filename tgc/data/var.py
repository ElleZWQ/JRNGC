import numpy as np
from ..runcase import data_decorator

def make_var_stationary(beta, radius=0.998):
    '''Rescale coefficients of VAR model to make stable.'''
    d = beta.shape[0]
    lag = beta.shape[2]
    top = np.hstack(
        (np.zeros((d * (lag - 1), d),
                  dtype=np.float32), np.eye(d * (lag - 1), dtype=np.float32)))
    bottom = np.transpose(beta, (0, 2, 1)).reshape(d, -1)
    beta_tilde = np.vstack((top, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.99 * beta, radius)
    else:
        return beta

@data_decorator
def var_stable(d, t, t_eval, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    beta = np.zeros((d, d, lag), dtype=np.float32)
    gc = np.zeros((d, d, lag), dtype=np.int32)

   
    vis = set()
    num_nonzero = int(d * d * sparsity)
    for _ in range(num_nonzero):
        i, j = np.random.choice(d, size=2, replace=False) # replace=False 表示不允许重复选择。因此，在这种情况下，得到的 i 和 j 会是不同的两个随机数。
        while (i, j) in vis:
            i, j = np.random.choice(d, size=2, replace=False)
        vis.add((i, j))
        tlag = np.random.randint(lag)
        beta[i, j, tlag] = beta_value * 2 * (np.random.randint(2) - 0.5)
        gc[i, j, tlag] = 1

    
    beta = make_var_stationary(beta)

   
    burn_in = 500
    errors = np.random.normal(scale=sd, size=(d, t + t_eval + burn_in))
    x = np.zeros((d, t + t_eval + burn_in))
    x[:, :lag] = errors[:, :lag]
    for t in range(lag, t + t_eval + burn_in):
        x[:, t] = np.einsum('jit,it->j', beta, x[:,t - lag:t]) + errors[:, t]
    
   

    x = x[:, burn_in:].astype(np.float32)
    m = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    x = (x - m) / sd
    return x[:, :t], x[:, :t], gc