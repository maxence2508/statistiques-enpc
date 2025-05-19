import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import geom

n=1000
p=0.1
x=np.arange(1,50)

# générer un échantillon de n va géométriques
sample=geom.rvs(p,size=n)

# création de l'histogramme
plt.hist(sample, density=True, bins=x, color='g', label='Sample Histogram')

# superposition de la fonction de masse théorique
plt.plot(x, geom.pmf(x,p), color='b', label='Theorical PMF')

# titre et étiquettes
plt.xlabel('value')
plt.ylabel('probability')
plt.title('Histogram of geometric ditribution')
plt.legend()

plt.show()