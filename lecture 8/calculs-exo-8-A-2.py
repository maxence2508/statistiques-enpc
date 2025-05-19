import scipy.stats as stats
import numpy as np

n = 25
theta = np.log(1.5)
alpha = 5/100
proba1 = stats.norm.cdf(stats.norm.ppf(1-alpha)-np.sqrt(n)*theta)
print(proba1)

proba2 = stats.norm.cdf(stats.norm.ppf(alpha)-np.sqrt(n)*theta)
print(proba2)