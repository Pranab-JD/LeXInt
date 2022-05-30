function [phi_1_array] = phi_1(z)
            
    phi_1_array = zeros(1, length(z));
    
    for ii = 1 : length(z)
        if abs(z(ii)) <= 1e-7
            phi_1_array(ii) = 1 + z(ii) * (1./2. + z(ii) * (1./6. + z(ii) * (1./24. + 1./120. * z(ii))));
        else
            phi_1_array(ii) = (exp(z(ii)) - 1)/z(ii);
        end
    end
end