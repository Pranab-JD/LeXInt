%%% -------------------------------------------------------------
%
% Created on Thu May 12 12:24:52 2022
%
% @author: Pranab JD
%
% Description: -
%         Functions to determine the largest eigenvalue of a 
%         matrix/related matrix.
%       
%         Gershgorin's disks can be used only if the matrix is 
%         explicitly available. For matrix-free implementation, 
%         choose power iterations.
%
%%% -------------------------------------------------------------

function [eig_real, eig_imag] = Gershgorin(A)
    %%% ---------------------------------------------------

    % Parameters
    % ----------
    % A        : N x N matrix

    % Returns
    % -------
    % eig_real : Largest real eigen value (negative magnitude)
    % eig_imag : Largest imaginary eigen value

    %%% ---------------------------------------------------

    %%% Divide matrix 'A' into Hermitian and skew-Hermitian
    A_Herm = (A + A')/2;
    A_SkewHerm = (A - A')/2;

    %%% # of rows/columns
    N = size(A, 1);

    row_sum_real = zeros(N, 1);
    row_sum_imag = zeros(N, 1);

    for ii = 1 : N
        row_sum_real(ii) = sum(abs(A_Herm(ii, :)));
        row_sum_imag(ii) = sum(abs(A_SkewHerm(ii, :)));
    end

    eig_real = - max(row_sum_real);       % Has to be NEGATIVE
    eig_imag = max(row_sum_imag);
    
end