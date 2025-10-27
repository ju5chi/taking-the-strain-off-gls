import numpy as np


### Numerical differentiation

def _diff(time, X):
        d_X = (X[1:] - X[:-1])/(time[1:, None] - time[:-1, None])
        d_time = time[:-1]

        return d_time, d_X

def _diff2(time, X):
        d_X_left = (X[1:-1] - X[:-2])/(time[1:-1, None] - time[:-2, None])
        d_X_right = (X[2:] - X[1:-1])/(time[2:, None] - time[1:-1, None])

        d_X = (d_X_left + d_X_right)/2
        d_time = time[1:-1]

        return d_time, d_X

def _diff3(time, X):
    d_X_left = (X[1:-1] - X[:-2])
    d_X_right = (X[2:] - X[:-2])

    d_X = (d_X_left + d_X_right/2)/2
    d_time = time[1:-1]

    return d_time, d_X

def differentiate(time, X, method='default'):
    if method == 'default':
        return _diff(time, X)
    
    elif method == 'symmetric':
        return _diff2(time, X)

    elif method == 'no_time_symmetric':
        return _diff3(time, X)
    
    else:
        raise ValueError(f'{method} is not a valid method.')



### Numerical integration

def _trapezoidal_rule(time, X):
    # use trapezoidal rule
    summation = 0.5 * (X[:-1] + X[1:]) * (time[1:] - time[:-1])
    return np.sum(summation)

def integrate(time, X, method='trapezoid'):
    if method == 'trapezoid':
        return _trapezoidal_rule(time, X)

    else:
        raise ValueError(f'{method} is not a valid method.')

def antiderivative(time, X, constant=0.0):
    antid_X = np.zeros(len(X)) + constant

    for i in range(len(X)):
        antid_X[i] = integrate(time[:i+1], X[:i+1])
    
    return antid_X
