import numpy as np

def A_tilde(A, B, v):
    """_summary_

    Args:
        A (function handle R^n -> R^n): _description_
        B (matrix, n*p): _description_
        v (vector, n+p): _description_

    Returns:
        y (1D vector) : \tilde{A} v with \tilde{A} = [A B; 0 K] and K = [0 I; 0 0]
    """

    [p, n] = np.shape(B)
    
    # print(np.shape(A(v[0:n]).reshape(1, n)))
    # print(np.shape(B))
    # print(np.shape(v[n:n+p]))
    
    y = np.concatenate([    A(v[0:n]).reshape(1, n) + B*v[n:n+p],
                            [v[n+1:n+p]],
                            np.array([0]).reshape(1, 1)], axis = 1)
    
    return y.reshape(np.shape(y)[1])