import numpy as np
from scipy.integrate import odeint
from ..runcase import data_decorator

def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F
    return dxdt

@data_decorator
def lorenz_96(d, t, t_eval, f, seed, delta_t=0.1, sd=0.1, burn_in=1000):
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=d)
    tm = np.linspace(0, (t + t_eval + burn_in) * delta_t, t + t_eval + burn_in)
    x = odeint(lorenz, x0, tm, args=(f, ))
    x += np.random.normal(scale=sd, size=(t + t_eval + burn_in, d))

    maxlag = 1
    # Set up Granger causality ground truth.
    gc = np.zeros((d, d, maxlag), dtype=int)
    for i in range(d):
        gc[i, i, maxlag - 1] = 1
        gc[i, (i + 1) % d, maxlag - 1] = 1
        gc[i, (i - 1) % d, maxlag - 1] = 1
        gc[i, (i - 2) % d, maxlag - 1] = 1

    x = np.swapaxes(x[burn_in:].astype(np.float32), 0, 1)
    m = np.mean(x, axis=1, keepdims=True)
    sd = np.std(x, axis=1, keepdims=True)
    x = (x - m) / sd
    return x[:, :t], x[:, t:], gc