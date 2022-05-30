function [u_epirk3, u_epirk4, num_rhs_calls] = EPIRK4s3A(u, dt, RHS_func, c, Gamma, rel_tol)
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
    % Real_Imag_Leja  : 0 - Real, 1 - Imaginary

    % Returns
    % -------
    % u_epirk3        : 1D vector u (output) after time dt (3rd order)
    % u_epirk4        : 1D vector u (output) after time dt (4th order)
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

        u_epirk3 = u;
        u_epirk4 = u + (2*rel_tol*u);
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

    %%% Internal stage 2; b = u + 2/3 phi_1(2/3 J(u) dt) f(u) dt
    [b_n_f, rhs_calls_3, ~] = real_Leja_phi(u, 2*dt/3, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);
    b_n = u + (b_n_f * 2*dt/3);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon;

    %%% F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b;

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% 3rd & 4th order solution
    [u_nl_3,  rhs_calls_4, ~] = real_Leja_phi(u, dt, RHS_func, 8*(Nonlin_a - Nonlin_u), c, Gamma, @phi_3, rel_tol);
    [u_nl_4a, rhs_calls_5, ~] = real_Leja_phi(u, dt, RHS_func, (-37/2*Nonlin_u + 32*Nonlin_a - 27/2*Nonlin_b), c, Gamma, @phi_3, rel_tol);
    [u_nl_4b, rhs_calls_6, ~] = real_Leja_phi(u, dt, RHS_func, (63*Nonlin_u - 144*Nonlin_a + 81*Nonlin_b), c, Gamma, @phi_4, rel_tol);

    %%% u_3 = u + phi_1(J(u) dt) f(u) dt + 8 phi_3(J(u) dt) (F(a) - F(u)) dt
    u_epirk3 = u + (u_flux * dt) + (u_nl_3 * dt);

    %%% u_4 = u + phi_1(J(u) dt) f(u) dt + phi_3(J(u) dt) (-37/2F(u) + 32F(a) - 27/2F(b)) dt
    %%%                                  + phi_4(J(u) dt) (63F(u) - 144F(a) + 81F(b)) dt
    u_epirk4 = u + (u_flux * dt) + (u_nl_4a * dt) + (u_nl_4b * dt);

    %%% Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 + rhs_calls_5 + rhs_calls_6 + 6;

end
