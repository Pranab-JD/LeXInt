%%% -------------------------------------------------------------
%
% Created on Thu May 12 12:24:52 2022
%
% @author: Pranab JD
%
% Description: -
%         Function to determine the largest eigenvalue of a 
%         matrix/related matrix.
%       
%         Gershgorin's disks can be used only if the matrix is 
%         explicitly available. For matrix-free or Jacobian-free
%         implementation, choose power iterations.
%
%%% -------------------------------------------------------------

function [largest_eigen_value, num_rhs_calls] = Power_iteration(u, RHS_function)
    %%% ---------------------------------------------------

    % Parameters
    % ----------
    % u                       : 1D vector u (input)
    % RHS_func	              : RHS function

    % Returns
    % -------
    % largest_eigen_value     : Largest eigen value (within 10% accuracy)
    % 2*(ii - 1)              : # of RHS calls

    %%% ---------------------------------------------------

    tol = 0.1;                                       % 10% tolerance
    niters = 1000;                                   % Max. # of iterations
    epsilon = 1e-7;
    eigen_value = zeros(1, niters);                  % Array of max. eigen value at each iteration
    vector = zeros(length(u), 1); vector(1) = 1;     % Initial estimate of eigen vector

    for ii = 2 : niters

        %%% Compute new eigen vector
        eigen_vector = (RHS_function(u + (epsilon * vector)) - RHS_function(u))/epsilon;

        %%% Max of eigen vector = eigen value
        eigen_value(ii) = max(abs(eigen_vector));

        %%% Convergence is to be checked for eigen values, not eigen vectors
        %%% since eigen values converge faster than eigen vectors
        if (abs(eigen_value(ii) - eigen_value(ii - 1)) <= tol * eigen_value(ii))
            
            largest_eigen_value = - eigen_value(ii);           % Real eigen value has to be NEGATIVE
            num_rhs_calls = 2*(ii - 1);
            
            break
        else
            eigen_vector = eigen_vector/eigen_value(ii);       % Normalize eigen vector to eigen value
            vector = eigen_vector;                             % New estimate of eigen vector
        end

    end    

end