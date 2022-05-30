function [phi_4_array] = phi_4(z)
            
    phi_4_array = zeros(1, length(z));
    
    for ii = 1 : length(z)
        if abs(z(ii)) <= 1e-5
            phi_4_array(ii) = 1./24. + z(ii) * (1./120. + z(ii) * (1./720. + z(ii) * (1./5040. + 1./40320. * z(ii))));
        else
            phi_4_array(ii) = (exp(z(ii)) - z(ii)^3/6 - z(ii)^2/2 - z(ii) - 1)/z(ii)^4;
        end
    end
end