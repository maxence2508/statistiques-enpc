import numpy as np
from scipy.stats import norm

# Échantillon de Z'
def sample_from_z(n):
    # Génération uniforme et transformation en quantiles de la normale
    u = np.random.uniform(0, 1, n)
    y = norm.ppf(np.sort(u))  # Utilisation du tri direct sur u pour obtenir y

    moy = np.mean(y)
    std = np.std(y)

    # Calcul de la statistique m de manière vectorisée
    cdf_values = norm.cdf((y - moy) / std)
    empirical_cdf = np.arange(0, n) / n
    empirical_cdf_2 = np.arange(1, n+1) / n
    a = np.max(np.abs(empirical_cdf - cdf_values))
    b = np.max(np.abs(empirical_cdf_2 - cdf_values))
    m = max(a,b)
    return m

# Simulations de Monte-Carlo
N_sim = 1000  # Nombre de simulations
stat_value = 0.05
n = 1000  # Taille des échantillons

# Lancement des simulations
samples = np.array([sample_from_z(n) for _ in range(N_sim)])

# Calcul de la p-valeur
p_value = np.mean(samples >= stat_value)
print(p_value)