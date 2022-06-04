function [u_exprb3, u_exprb5, num_rhs_calls] = EXPRB53s3(u, dt, RHS_func, c, Gamma, rel_tol)
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
    % u_exprb3        : 1D vector u (output) after time dt (3rd order)
    % u_exprb5        : 1D vector u (output) after time dt (5th order)
    % num_rhs_calls   : # of RHS calls
    %
    %%% ---------------------------------------------------
    
    %%% RHS of PDE at u
    f_u = RHS_func(u);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Check if iterations converge for largest value of "f_u" and "dt".
    %%% If this stage doesn't converge, we have to go for a smaller 
    %%% step size. "u_flux" is used in the final stage.
    
    [u_flux, rhs_calls_1, convergence] = real_Leja_phi(u, dt, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);

    %%% If it does not converge, return (try with smaller dt)
    if convergence == 0

        u_exprb3 = u;
        u_exprb5 = u + (2*rel_tol*u);
        num_rhs_calls = rhs_calls_1;
        
        return
    end

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Internal stage 1; a = u + 1/2 phi_1(1/2 J(u) dt) f(u) dt
    [a_n_f, rhs_calls_2, ~] = real_Leja_phi(u, dt/2, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);
    a_n = u + (a_n_f * dt/2);

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

    %%% Internal stage 2; b = u + 9/10 phi_1(9/10 J(u) dt) f(u) dt + (27/25 phi_3(1/2 J(u) dt) + 729/125 phi_3(9/10 J(u) dt)) (F(a) - F(u)) dt
    [b_n_f,    rhs_calls_3, ~] = real_Leja_phi(u, 9*dt/10, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);
    [b_n_nl_1, rhs_calls_4, ~] = real_Leja_phi(u, dt/2, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, @phi_3, rel_tol);
    [b_n_nl_2, rhs_calls_5, ~] = real_Leja_phi(u, 9*dt/10, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, @phi_3, rel_tol);

    b_n = u + (9/10 * b_n_f * dt) + (27/25 * b_n_nl_1 * dt) + (729/125 * b_n_nl_2 * dt);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon;

    %%% F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b;

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% 3rd & 5th order solution
    [u_nl_3,  rhs_calls_6, ~] = real_Leja_phi(u, dt, RHS_func, (-(312/81)*Nonlin_u + 2*Nonlin_a + (150/81)*Nonlin_b), c, Gamma, @phi_3, rel_tol);
    [u_nl_5a, rhs_calls_7, ~] = real_Leja_phi(u, dt, RHS_func, (-(1208/81)*Nonlin_u + 18*Nonlin_a - (250/81)*Nonlin_b), c, Gamma, @phi_3, rel_tol);
    [u_nl_5b, rhs_calls_8, ~] = real_Leja_phi(u, dt, RHS_func, ((1120/27)*Nonlin_u - 60*Nonlin_a + (500/27)*Nonlin_b), c, Gamma, @phi_4, rel_tol);

    %%% u_3 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) ((-312/81)F(u) + 2F(a) + (150/81)F(b)) dt
    u_exprb3 = u + (u_flux * dt) + (u_nl_3 * dt);

    %%% u_5 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) ((-1208/81)F(u) + 18F(a) - (250/81)F(b)) dt + phi_4(J(u) dt) ((1120/27)F(u) - 60F(a) + (500/27)F(b)) dt
    u_exprb5 = u + (u_flux * dt) + (u_nl_5a * dt) + (u_nl_5b * dt);

    %%% Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + ...
                    rhs_calls_6 + rhs_calls_7 + rhs_calls_8 + 6;

end
