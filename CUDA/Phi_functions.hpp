#pragma once

#include "functions.hpp"

//? Phi Functions ('z' is assumed to a double)

double phi_1(double z)
{    
    double phi_1_value;
    
    if (abs(z) <= 1e-7)
    {
        phi_1_value = 1./factorial(1) + z * (1./factorial(2)  + z * (1./factorial(3) + \
                                        z * (1./factorial(4)  + z * (1./factorial(5) + \
                                        z * (1./factorial(6)  + z * (1./factorial(7) + \
                                        z * (1./factorial(8)  + z * (1./factorial(9) + \
                                        z * (1./factorial(10) + z * (1./factorial(11)))))))))));
    }   
    else
    {
        phi_1_value = (exp(z) - 1)/z;
    }
            
    return phi_1_value;
}


double phi_2(double z)
{    
    double phi_2_array;
    
    if (abs(z) <= 1e-6)
    {
        phi_2_array = 1./factorial(2) + z * (1./factorial(3)  + z * (1./factorial(4)  + \
                                        z * (1./factorial(5)  + z * (1./factorial(6)  + \
                                        z * (1./factorial(7)  + z * (1./factorial(8)  + \
                                        z * (1./factorial(9)  + z * (1./factorial(10) + \
                                        z * (1./factorial(11) + z * (1./factorial(12)))))))))));
    }
    else
    {
        phi_2_array = (exp(z) - z - 1)/(z*z);
    }
            
    return phi_2_array;
}


double phi_3(double z)
{    
    double phi_3_array;
    
    if (abs(z) <= 1e-5)
    {
        phi_3_array = 1./factorial(3) + z * (1./factorial(4)  + z * (1./factorial(5) + \
                                        z * (1./factorial(6)  + z * (1./factorial(7) + \
                                        z * (1./factorial(8)  + z * (1./factorial(9) + \
                                        z * (1./factorial(10) + z * (1./factorial(11) + \
                                        z * (1./factorial(12) + z * (1./factorial(13)))))))))));
    }
    else
    {
        phi_3_array = (exp(z) - (z*z)/2 - z - 1)/(z*z*z);
    }
            
    return phi_3_array;
}


double phi_4(double z)
{
    double phi_4_array;
    
    if (abs(z) <= 1e-4)
    {
        phi_4_array = 1./factorial(4) + z * (1./factorial(5)  + z * (1./factorial(6)  + \
                                        z * (1./factorial(7)  + z * (1./factorial(8)  + \
                                        z * (1./factorial(9)  + z * (1./factorial(10) + \
                                        z * (1./factorial(11) + z * (1./factorial(12) + \
                                        z * (1./factorial(13) + z * (1./factorial(14)))))))))));     
    }
    else
    {
        phi_4_array = (exp(z) - (z*z*z)/6 - (z*z)/2 - z - 1)/(z*z*z*z);
    }
            
    return phi_4_array;
}