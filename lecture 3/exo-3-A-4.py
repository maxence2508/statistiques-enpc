import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Définition des fonctions
def r_H(alpha):
    return np.sqrt((-1/2) * np.log(alpha / 2))

def r_BC_tilde(alpha):
    return 1 / np.sqrt(18 * alpha)

# Calculer l'intersection des courbes
def difference(alpha):
    return r_H(alpha) - r_BC_tilde(alpha)

# Utilisation de fsolve pour trouver le point où les deux courbes se croisent
alpha_intersection = fsolve(difference, 0.001)[0]  # estimation initiale proche de 0.1

# Calculer les valeurs correspondantes des fonctions au point d'intersection
r_H_intersection = r_H(alpha_intersection)
r_BC_tilde_intersection = r_BC_tilde(alpha_intersection)

# Générer une plage de valeurs pour alpha sur [0, 1]
alpha_values = np.linspace(0.001, 1, 400)

# Calculer les valeurs des deux fonctions
r_H_values = r_H(alpha_values)
r_BC_tilde_values = r_BC_tilde(alpha_values)

# Tracer les deux fonctions
plt.figure(figsize=(8,6))
plt.plot(alpha_values, r_H_values, label=r'$r^H(\alpha)$', color='b')
plt.plot(alpha_values, r_BC_tilde_values, label=r'$\tilde{r}^{BC}(\alpha)$', color='r')

# Ajouter une ligne verticale à l'endroit de l'intersection
plt.axvline(x=alpha_intersection, color='g', linestyle='--', label=f'Intersection: $\\alpha={alpha_intersection:.3f}$')

# Ajouter des points à l'endroit de l'intersection
plt.scatter(alpha_intersection, r_H_intersection, color='black', zorder=5)
plt.scatter(alpha_intersection, r_BC_tilde_intersection, color='black', zorder=5)

# Annoter les points d'intersection
plt.annotate(f'({alpha_intersection:.3f}, {r_H_intersection:.3f})',
             xy=(alpha_intersection, r_H_intersection),
             xytext=(alpha_intersection + 0.05, r_H_intersection - 0.1),
             arrowprops=dict(facecolor='black', arrowstyle='->'))

# Ajouter des labels et une légende
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$r$')
plt.title('Comparaison des fonctions r^H(a) et r_tilde^BC(a) sur [0, 1]')
plt.legend()

# Ajouter une grille
plt.grid(True)

# Afficher le graphe
plt.show()