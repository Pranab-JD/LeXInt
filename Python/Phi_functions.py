import numpy as np

### Phi Functions ('z' is assumed to be an array of doubles or complex doubles)

def phi_1(z):
    
    if np.imag(z[0]) != 0.0:
        phi_1_array = np.zeros(len(z), dtype = "complex")
    else:
        phi_1_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-7:
            phi_1_array[ii] = 1./np.math.factorial(1) + z[ii] * (1./np.math.factorial(2)  + z[ii] * (1./np.math.factorial(3) + \
                                                        z[ii] * (1./np.math.factorial(4)  + z[ii] * (1./np.math.factorial(5) + \
                                                        z[ii] * (1./np.math.factorial(6)  + z[ii] * (1./np.math.factorial(7) + \
                                                        z[ii] * (1./np.math.factorial(8)  + z[ii] * (1./np.math.factorial(9) + \
                                                        z[ii] * (1./np.math.factorial(10) + z[ii] * (1./np.math.factorial(11)))))))))))     
        else:
            phi_1_array[ii] = (np.exp(z[ii]) - 1)/z[ii]
            
    return phi_1_array


def phi_2(z):
    
    if np.imag(z[0]) != 0.0:
        phi_2_array = np.zeros(len(z), dtype = "complex")
    else:
        phi_2_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-6:
            phi_2_array[ii] = 1./np.math.factorial(2) + z[ii] * (1./np.math.factorial(3)  + z[ii] * (1./np.math.factorial(4)  + \
                                                        z[ii] * (1./np.math.factorial(5)  + z[ii] * (1./np.math.factorial(6)  + \
                                                        z[ii] * (1./np.math.factorial(7)  + z[ii] * (1./np.math.factorial(8)  + \
                                                        z[ii] * (1./np.math.factorial(9)  + z[ii] * (1./np.math.factorial(10) + \
                                                        z[ii] * (1./np.math.factorial(11) + z[ii] * (1./np.math.factorial(12)))))))))))     
        else:
            phi_2_array[ii] = (np.exp(z[ii]) - z[ii] - 1)/z[ii]**2
        
    return phi_2_array


def phi_3(z):
    
    if np.imag(z[0]) != 0.0:
        phi_3_array = np.zeros(len(z), dtype = "complex")
    else:
        phi_3_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-5:
            phi_3_array[ii] = 1./np.math.factorial(3) + z[ii] * (1./np.math.factorial(4)  + z[ii] * (1./np.math.factorial(5)  + \
                                                        z[ii] * (1./np.math.factorial(6)  + z[ii] * (1./np.math.factorial(7)  + \
                                                        z[ii] * (1./np.math.factorial(8)  + z[ii] * (1./np.math.factorial(9)  + \
                                                        z[ii] * (1./np.math.factorial(10) + z[ii] * (1./np.math.factorial(11) + \
                                                        z[ii] * (1./np.math.factorial(12) + z[ii] * (1./np.math.factorial(13)))))))))))     
        else:
            phi_3_array[ii] = (np.exp(z[ii]) - z[ii]**2/2 - z[ii] - 1)/z[ii]**3
    
    return phi_3_array


def phi_4(z):
    
    if np.imag(z[0]) != 0.0:
        phi_4_array = np.zeros(len(z), dtype = "complex")
    else:
        phi_4_array = np.zeros(len(z))
    
    for ii in range(len(z)):
        if abs(z[ii]) <= 1e-3:
            phi_4_array[ii] = 1./np.math.factorial(4) + z[ii] * (1./np.math.factorial(5)  + z[ii] * (1./np.math.factorial(6)  + \
                                                        z[ii] * (1./np.math.factorial(7)  + z[ii] * (1./np.math.factorial(8)  + \
                                                        z[ii] * (1./np.math.factorial(9)  + z[ii] * (1./np.math.factorial(10) + \
                                                        z[ii] * (1./np.math.factorial(11) + z[ii] * (1./np.math.factorial(12) + \
                                                        z[ii] * (1./np.math.factorial(13) + z[ii] * (1./np.math.factorial(14)))))))))))        
        else:
            phi_4_array[ii] = (np.exp(z[ii]) - z[ii]**3/6 - z[ii]**2/2 - z[ii] - 1)/z[ii]**4
        
    return phi_4_array
