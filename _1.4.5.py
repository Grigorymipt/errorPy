import numpy as np
import matplotlib.pyplot as plt
import math

#input
m = [0] #fill
sigma_m = [0] #fill
l = 1 #fill
sigma_l = 0 #fill
#nu = [109, 218, 329, 440, 553, 668, 784] #fill T
#n = [10,20,30,40,50,60,70] #fill Q

files = ['TextFile1.txt']

for k in range(len(files)):
    input = open(files[k], 'r')
    lines = input.readlines()
    # print(lines)
    nu = [0] * len(lines)
    n = [0] * len(lines)
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip()
        lines[i] = lines[i].split(',')
        for j in range(2):
            lines[i][j] = float(lines[i][j])
    for i in range(len(lines)):
        nu[i] = lines[i][0] 
        n[i] = ((math.log(lines[i][1]/lines[0][1]))**2)**0.5 

plt.plot(n, nu)
plt.grid(True)
#plt.show()


k = min(len(nu), len(n))

#random error T = nu, Q = n (nu/n)
def rand_error(nu, n, t, k):
    if t == True:
        nu_2_rand = 0
        n_2_rand = 0
        nu_n_rand = 0
        nu_rand = 0
        n_rand = 0
        k = k//4
        for i in range(k):
            i = i*4 + 3
            nu_2_rand += nu[i]**2
            n_2_rand += n[i]**2
            nu_n_rand += nu[i]*n[i]
            nu_rand += nu[i]
            n_rand += n[i]
        nu_2_rand = nu_2_rand/(k)
        n_2_rand = n_2_rand/(k)
        nu_n_rand = nu_n_rand/(k) # -
        nu_rand = nu_rand/(k)
        n_rand = n_rand/(k) # -
        print(nu_2_rand, n_2_rand, nu_n_rand, nu_rand, n_rand)
        
        p, v = np.polyfit(n, nu, deg=1, cov=True)
        print(p[0])
        print(v)
        b = ((nu_n_rand - n_rand * nu_rand)/(n_2_rand - n_rand**2)) # wo n_2_rand
        #print(((nu_2_rand - nu_rand**2)/(n_2_rand - n_rand**2)), b**2)
        #print((nu_2_rand - nu_rand**2) - (nu_n_rand - n_rand * nu_rand), (n_2_rand - n_rand**2))
        bSigma = (k)**(-0.5)*((((nu_2_rand - nu_rand**2) - (nu_n_rand - n_rand * nu_rand)**2)/(n_2_rand - n_rand**2)))**0.5
        bSigma = (k)**(-0.5)*(( (nu_2_rand - nu_rand**2) - ((nu_n_rand - n_rand * nu_rand)**2)/(n_2_rand - n_rand**2))/(n_2_rand - n_rand**2) )**0.5
        a = nu_rand - b * n_rand
        aSigma = bSigma * ((n_2_rand - n_rand**2))**0.5
        return b, a, bSigma, aSigma
#systematic error
#TODO:
"""
def syst_error(sigmas, meas, main_m):
    epsilon = 0
    for i in range(min(len(sigmas, meas))):
        epsilon = (epsilon**2 + sigmas[i]**2)**0.5
    sigma = epsilon * main_m
    return sigma"""
"""
p =  np.polyfit(n, nu, deg = 1)
plt.plot()
plt.show()
"""
error = rand_error(nu, n, True, k)
print(error[0], error[2])
