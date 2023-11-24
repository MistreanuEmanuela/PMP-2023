import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import numpy as np


def main():
    np.random.seed(42)
    wait_time = np.random.normal(loc=10, scale=1, size=100)

# generam 100 de timpi de asteptare folosind o distributie normala intre 10 si 1
    print(np.mean(wait_time))
# media este 9.896153482605905
    with pm.Model() as model:
        miu = pm.Normal('miu', mu=10)
        sigma = pm.HalfNormal('sigma', sigma=1)
        likelihood = pm.Normal('likelihood', mu=miu, sigma=sigma, observed=wait_time)
        trace = pm.sample(100, tune=100)

#folosim distributia normala presupunand ca timpii de asteptare sunt aproximativ normali si de acceasi valoare
    with model:
        # az.plot_posterior(trace)
        az.plot_posterior(trace, var_names=['miu'])
        plt.title('Distributia pentru miu')
        plt.xlabel('Miu')
        plt.ylabel('Probabilitatea')
        plt.show()
#observam ca probabilitatea ca miu sa fie in intervalul 9.8, 10 este 94%, asadar aproximativ egala cu media
#calculata pentru wait_time generat random
if __name__=='__main__':
    main()


