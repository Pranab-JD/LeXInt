function [u_epirk4, u_epirk5, num_rhs_calls] = EPIRK5P1(u, dt, RHS_func, c, Gamma, rel_tol)
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

    %%% Parameters of EPIRK5P1 (5th order)
    a11 = 0.35129592695058193092;
    a21 = 0.84405472011657126298;
    a22 = 1.6905891609568963624;

    b1  = 1.0;
    b2  = 1.2727127317356892397;
    b3  = 2.2714599265422622275;

    g11 = 0.35129592695058193092;
    g21 = 0.84405472011657126298;
    g22 = 0.5;
    g31 = 1.0;
    g32 = 0.71111095364366870359;
    g33 = 0.62378111953371494809;

    %%% 4th order
    g32_4 = 0.5;
    g33_4 = 1.0;
    
    %%% RHS of PDE at u
    f_u = RHS_func(u);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Check if iterations converge for largest value of "f_u" and "dt".
    %%% If this stage doesn't converge, we have to go for a smaller 
    %%% step size. "u_flux" is used in the final stage.
    
    [u_flux, rhs_calls_1, convergence] = real_Leja_phi(u, dt, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);

    %%% If it does not converge, return (try with smaller dt)
    if convergence == 0

        u_epirk4 = u;
        u_epirk5 = u + (2*rel_tol*u);
        num_rhs_calls = rhs_calls_1;
        
        return
    end

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Internal stage 1 
    [a_n_f, rhs_calls_2, ~] = real_Leja_phi(u, g11*dt, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);
    a_n = u + (a11 * a_n_f * dt);

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

    %%% Internal stage 2
    [b_n_f,  rhs_calls_3, ~] = real_Leja_phi(u, g21*dt, RHS_func, f_u, c, Gamma, @phi_1, rel_tol);
    [b_n_nl, rhs_calls_4, ~] = real_Leja_phi(u, g22*dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, @phi_1, rel_tol);
    b_n = u + (a21 * b_n_f * dt) + (a22 * b_n_nl * dt);

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% J(u) * b
    Linear_b = (RHS_func(u + (epsilon * b_n)) - f_u)/epsilon;

    %%% F(b) = f(b) - (J(u) * b)
    Nonlin_b = RHS_func(b_n) - Linear_b;

    %%%%%%%%%%%%%%%%%% --------------------- %%%%%%%%%%%%%%%%%%

    %%% Nonlinear remainders for 4th order solution
    [u_nl_4a, rhs_calls_5, ~] = real_Leja_phi(u, g32_4*dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, @phi_1, rel_tol);
    [u_nl_4b, rhs_calls_6, ~] = real_Leja_phi(u, g33_4*dt, RHS_func, (Nonlin_u - 2*Nonlin_a + Nonlin_b), c, Gamma, @phi_3, rel_tol);

    %%% Nonlinear remainders for 5th order solution
    [u_nl_5a, rhs_calls_7, ~] = real_Leja_phi(u, g32*dt, RHS_func, (Nonlin_a - Nonlin_u), c, Gamma, @phi_1, rel_tol);
    [u_nl_5b, rhs_calls_8, ~] = real_Leja_phi(u, g33*dt, RHS_func, (Nonlin_u - 2*Nonlin_a + Nonlin_b), c, Gamma, @phi_3, rel_tol);
    
    %%% 4th and 5th order solutions
    u_epirk4 = u + (u_flux * b1 * dt) + (u_nl_4a * b2 * dt) + (u_nl_4b * b3 *dt);
    u_epirk5 = u + (u_flux * b1 * dt) + (u_nl_5a * b2 * dt) + (u_nl_5b * b3 *dt);

    %%% Proxy of computational cost
    num_rhs_calls = rhs_calls_1 + rhs_calls_2 + rhs_calls_3 + rhs_calls_4 ...
                  + rhs_calls_5 + rhs_calls_6 + rhs_calls_7 + rhs_calls_8 + 6;

end
