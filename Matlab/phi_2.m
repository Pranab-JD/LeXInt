function [phi_2_array] = phi_2(z)
            
    phi_2_array = zeros(1, length(z));
    
    for ii = 1 : length(z)
        if abs(z(ii)) <= 1e-7
            phi_2_array(ii) = 1./2. + z(ii) * (1./6. + z(ii) * (1./24. + z(ii) * (1./120. + 1./720. * z(ii))));
        else
            phi_2_array(ii) = (exp(z(ii)) - z(ii) - 1)/z(ii)^2;
        end
    end
end