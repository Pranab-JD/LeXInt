import numpy as np

def A_tilde(A, B, v):
    """
    Input:
    A : function handle R^n -> R^n
    B : matrix (n x p)
    v : vector (n + p)

    Output:
    y = \tilde{A} v with \tilde{A} = [A B; 0 K] and K = [0 I; 0 0]
    """

    [n, p] = np.shape(B)

    y = [
            A(v[1:n]) + B*v[n+1:n+p]
            v[n+2:n+p]
            0
        ]
    return y


def linear_phi_Leja(u, Jacobian_vector, c, Gamma, Leja_X, tol, poly_coeffs):
    """
    % Evaluates a linear combinaton of the phi functions
    % evaluated at A acting on vectors from u:
    %
    % w = phi_0(A) u(:, 1) + phi_1(A) u(:, 2) + ... + phi_p(A) u(:, p+1)
    %
    % Inputs:
    % u                 : matrix of size (n) x (p+1)
    % dt                : Step size
    % J                 : Jacobian of RHS function
    % c                 : Shifting factor
    % Gamma             : Scaling factor
    % Leja_X            : Array of Leja points
    % tol               : Accuracy of the polynomial so formed
    %
    % Outputs:
    % w                 : linear combinaton of the phi functions (cf above)
    % convergence       : 0 - did not converge, 1 - converged
    %
    """

    B = np.fliplr(u[:, 2:-1]);
    [n, m] = np.shape(u);
    p = m - 1;

    Atx = lambda x: A_tilde(Jacobian_vector, B, x);
    v = [u[:,1]; np.zeros((p-1, 1)); 1];

    [polynomial, convergence] = real_Leja_exp(v, Atx, c, Gamma, Leja_X, tol, poly_coeffs);

    return polynomial(1 : n), convergence
