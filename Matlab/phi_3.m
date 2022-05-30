function [phi_3_array] = phi_3(z)
            
    phi_3_array = zeros(1, length(z));
    
    for ii = 1 : length(z)
        if abs(z(ii)) <= 1e-6
            phi_3_array(ii) = 1./6. + z(ii) * (1./24. + z(ii) * (1./120. + z(ii) * (1./720. + 1./5040. * z(ii))));
        else
            phi_3_array(ii) = (exp(z(ii)) - z(ii)^2/2 - z(ii) - 1)/z(ii)^3;
        end
    end
end