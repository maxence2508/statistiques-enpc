import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson

# Set the true value of theta
theta_true = 9

# Choose two sets of hyperparameters (a, lambda)
a1, lambda1 = 2, 1
a2, lambda2 = 3, 1

# Define the function to generate the Gamma prior
def gamma_prior(a, lambda_, theta_vals):
    return gamma.pdf(theta_vals, a, scale=1/lambda_)

# Set the range for theta values
theta_vals = np.linspace(0, 15, 500)

# Plot the prior distributions
plt.figure(figsize=(10, 6))
plt.plot(theta_vals, gamma_prior(a1, lambda1, theta_vals), label=r'$\Gamma(a=2, \lambda=1)$')
plt.plot(theta_vals, gamma_prior(a2, lambda2, theta_vals), label=r'$\Gamma(a=3, \lambda=1)$')
plt.title('Gamma Prior Distributions')
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Generate a sample of independent Poisson variables (X1, ..., XN)
N = 500  # Total number of samples
sample_size = 500  # Sample size to incrementally increase

X = poisson.rvs(theta_true, size=N)  # Poisson samples with theta_true

# Function to plot posterior for given (a, lambda)
def gamma_posterior(a, lambda_, X_data, n):
    posterior_a = a + np.sum(X_data[:n])  # sum of first n samples
    posterior_lambda = lambda_ + n
    posterior_vals = gamma.pdf(theta_vals, posterior_a, scale=1/posterior_lambda)
    return posterior_vals

# Plot the posterior distributions for increasing n
plt.figure(figsize=(12, 8))

for n in range(1, N+1, 50):  # Plot at every 10th sample
    posterior1 = gamma_posterior(a1, lambda1, X, n)
    posterior2 = gamma_posterior(a2, lambda2, X, n)

    plt.plot(theta_vals, posterior1, label=f'Posterior 1 (n={n})', color='blue', alpha=0.7)
    plt.plot(theta_vals, posterior2, label=f'Posterior 2 (n={n})', color='red', alpha=0.7)

plt.title('Posterior Distributions')
plt.xlabel(r'$\theta$')
plt.ylabel('Density')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()