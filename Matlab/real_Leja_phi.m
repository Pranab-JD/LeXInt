function [polynomial, num_rhs_calls, Leja_convergence] = real_Leja_phi(u, dt, RHS_function, interp_func, c, Gamma, phi_function, rel_tol)
    %%% ---------------------------------------------------
    %
    % Parameters
    % ----------
    % u                       : 1D vector u (input)
    % dt                      : Step size
    % RHS_function            : RHS function
    % interp_func             : function to be multiplied to phi function
    % c                       : Shifting factor
    % Gamma                   : Scaling factor
    % phi_function            : phi function
    % rel_tol                 : Accuracy of the polynomial so formed

    % Returns
    % ----------
    % polynomial              : Polynomial interpolation of 'interp_func' 
    %                           multiplied by 'phi_func' at real Leja points
    % num_rhs_calls           : # of RHS calls
    % Leja_convergence        : 0 - did not converge, 1 - converged
    %
    %%% ---------------------------------------------------

    %%% Phi function applied to 'interp_func' (scaled and shifted)
    function phi_function_array = func(xx)

        zz = (dt * (c + Gamma*xx));
        phi_function_array = phi_function(zz);

    end

    %%% Array of Leja points
    Leja_file = fopen('Leja_points.txt', 'r');
    Leja_X = fscanf(Leja_file, '%f');

    %%% Compute the polynomial coefficients
    coeffs = Divided_Difference(@func, Leja_X);

    %%% ------------------------------------------------------------------- %%%

    %%% a_0 term (form the polynomial)
    poly = interp_func;
    poly = coeffs(1) * poly;

    %%% ------------------------------------------------------------------- %%%

    %%% a_1, a_2 .... a_n terms
    max_Leja_pts = 500;                                      % Max number of Leja points
    y = interp_func;                                         % x values of the polynomial
    epsilon = 1e-7;
    Leja_convergence = 0;                                    % 0 - did not converge, 1 - converged

    %%% Iterate until converges
    for ii = 2 : max_Leja_pts

        %%% Compute the numerical Jacobian
        Jacobian_function = (RHS_function(u + (epsilon * y)) - RHS_function(u))/epsilon;

        %%% Re-scale and re-shift
        y = y * (-c/Gamma - Leja_X(ii - 1));
        y = y + (Jacobian_function/Gamma);

        %%% Approx. error incurred (accuracy)
        poly_error = sqrt(sum(abs(y).^2))/length(y) * abs(coeffs(ii));

        %%% If new number (next order) to be added < tol, break loop
        if  poly_error < rel_tol
%             disp('No. of Leja points used (real phi) = '), ii
%             disp('----------Tolerance reached---------------')
    
            poly = poly + (coeffs(ii) * y);
            Leja_convergence = 1;

            break

        %%% To stop diverging
        elseif poly_error > 1e7
%             disp('Starts to diverge after ", ii, " iterations.')
            polynomial = interp_func;
            num_rhs_calls = 2*ii; 
            Leja_convergence = 0;

            return

        else
           poly = poly + (coeffs(ii) * y);
        end
    end

    %%% ------------------------------------------------------------------- %%%

    %%% Solution
    polynomial = poly;
    num_rhs_calls = 2*ii;

end