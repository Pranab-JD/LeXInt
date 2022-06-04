function [u_exprb2, u_exprb3, num_rhs_calls] = EXPRB32(u, dt, RHS_func, c, Gamma, rel_tol)
    %%% ---------------------------------------------------
    %
    % Parameters
    % ----------
	% u               : 1D vector u (input)
	% dt              : Step size
	% RHS_func	      : RHS function
    % c               : Shifting factor
    % Gamma           : Scaling factor
    % rel_tol         : Accuracy of the polynomial so formed

    % Returns
    % -------
    % u_exprb2        : 1D vector u (output) after time dt (2nd order)
    % u_exprb3        : 1D vector u (output) after time dt (3rd order)
    % num_rhs_calls   : # of RHS calls
    %
    %%% ---------------------------------------------------
    
    %%% RHS of PDE at u
    f_u = RHS_func(u);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Internal stage 1 
    [a_n_f, rhs_calls_1, convergence] = real_Leja_phi(u, dt, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);

    %%% If it does not converge, return (try with smaller dt)
    if convergence == 0

        u_exprb2 = u;
        u_exprb3 = u + (2*rel_tol*u);
        num_rhs_calls = rhs_calls_1;
        
        return
    end

    a_n = u + (a_n_f * dt);

    %%% 2nd order solution
    u_exprb2 = a_n;

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    epsilon = 1e-7;

    %%% J(u) * u
    Linear_u = (RHS_func(u + (epsilon * u)) - f_u)/epsilon;

    %%% F(u) = f(u) - (J(u) * u)
    Nonlin_u = f_u - Linear_u;

    %%% J(u) * a
    Linear_a = (RHS_func(u + (epsilon * a_n)) - f_u)/epsilon;

    %%% F(a) = f(a) - (J(u) * a)
    Nonlin_a = RHS_func(a_n) - Linear_a;

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% 3rd order solution
    [u_3, rhs_calls_2, ~] = real_Leja_phi(u, dt, RHS_func, 2*(Nonlin_a - Nonlin_u), c, Gamma, @phi_3, rel_tol);

    u_exprb3 = u_exprb2 + (u_3 * dt);

    %%% Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + 4;

end
