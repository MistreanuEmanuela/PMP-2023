import numpy as np
from scipy.stats import poisson, norm, expon
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

lambda_v = 20
plasare_comanda = 2
deviatie_comanda = 0.5
alpha_v = 3
num_simulations = 100

average_wait_times = []

for _ in range(num_simulations):
    nr_clienti = poisson.rvs(mu=lambda_v)
    timp_comanda = norm.rvs(loc=plasare_comanda, scale=deviatie_comanda, size=nr_clienti)
    timp_pregatire = expon.rvs(scale=alpha_v, size=nr_clienti)
    timp_total = timp_comanda + timp_pregatire
    total_wait_time = np.sum(timp_total)
    average_wait_time = total_wait_time / nr_clienti
    average_wait_times.append(average_wait_time)

print("Un eșantion de 100 de timpi de așteptare medii:")
for i, time in enumerate(average_wait_times):
    print(f"Eșantion {i + 1}: {time:.2f} minute")

observed_data = np.array(average_wait_times)

with pm.Model() as model:
    alpha = pm.Exponential("alpha", lam=1/3)
    waiting_time = pm.Normal('waiting_time', mu=alpha, sd=deviatie_comanda, observed=observed_data)
    trace = pm.sample(2000, tune=100, cores=1)

az.plot_posterior(trace, var_names=['alpha'], kind='kde')
plt.title("Distribuția α")
plt.xlabel("α")
plt.show()

alpha_mean = np.mean(trace['alpha'])

print(f"Media de asteptare obtinuta: {alpha_mean:.2f}")
print(f"Media de asteptare propusa: 3")
