import numpy as np
import scipy.optimize
import scipy.integrate
import scipy.stats
import math

# Définir la fonction de densité gaussienne
def gaussian_density(x):
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x**2)

# Définir la fonction de répartition cumulative en utilisant scipy.integrate.quad
def F(x):
    result, _ = scipy.integrate.quad(gaussian_density, -np.inf, x)
    return result

# Définir la réciproque de la fonction de répartition
def F_inverse(p):
    # Définir la fonction dont nous voulons trouver la racine
    def func(x):
        return F(x) - p

    # Utiliser la méthode de Newton pour trouver la racine
    return scipy.optimize.newton(func, 0)

# Question 2

p0 = 0.18
n = 100
moy_emp = 0.16
beta = np.sqrt(n/(p0*(1-p0)))*(moy_emp-p0)
print("beta = ",beta)
p_valeur = F(beta)
print("p-valeur = ",p_valeur)

# Question 3

alpha = 5/100
s = F_inverse(alpha)
print("F_G^-1(a) = ", s)

def p_n(k,prob):
    t = k*p0+np.sqrt(k*p0*(1-p0))*s
    return scipy.stats.binom.cdf(math.floor(t), k, prob)

p = (1-1/3)*p0

n_min = 1
while(p_n(n_min,p)<0.8):
    n_min+=1
print(n_min)


