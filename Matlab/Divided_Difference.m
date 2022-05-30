function [div_diff] = Divided_Difference(func, X)
    %%% ---------------------------------------------------

    % Parameters
    % ----------
    % X       : Leja points
    % func    : func(X)

    % Returns
    % -------
    % div_diff : Polynomial coefficients

    %%% ---------------------------------------------------

    N = length(X) - 1;
    div_diff = func(X);

    for jj = 1 : N
        for ii = N + 1 : -1 : jj + 1
            div_diff(ii) = (div_diff(ii) - div_diff(ii - 1))/(X(ii) - X(ii - jj)); 
        end
    end

end