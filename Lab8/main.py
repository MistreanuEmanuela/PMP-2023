import pandas as pd
import pymc3 as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

data = pd.read_csv('Prices.csv')
def main():
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        sigma = pm.HalfCauchy('sigma', 5)
        mu = pm.Deterministic('mu', alpha + beta1 * data['Speed'] + beta2 * np.log(data['HardDrive']))
        y = pm.Normal('y', mu=mu, sd=sigma, observed=data['Price'])
        trace = pm.sample(10, tune=10)

    az.plot_trace(trace, var_names=['alpha', 'beta1', 'beta2', 'sigma'])
    plt.show()

    hdi_beta1 = pm.stats.hdi(trace['beta1'], hdi_prob=0.95)
    hdi_beta2 = pm.stats.hdi(trace['beta2'], hdi_prob=0.95)

    print(f"Estimarea HDI pentru beta1: {hdi_beta1}")
    print(f"Estimarea HDI pentru beta2: {hdi_beta2}")

    if 0 in hdi_beta1:
        print("Nu avem suficienta informatie sa stim ca Frecventa poate influenta ( cum beta1 poate fi 0)")
    else:
        print("Frecventa influenteaza sigur/ este un predictor sigur aal pretului de vanzare ")

    if 0 in hdi_beta2:
        print("Nu avem suficienta informatie sa stim ca Marimea Hard Disk-ului poate influenta ( cum beta1 poate fi 0)")
    else:
        print("Marimea Hard Disk-ului influenteaza sigur/ este un predictor sigur aal pretului de vanzare ")

     # ex4+ ex5
    with pm.Model() as model1:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta1 = pm.Normal('beta1', mu=0, sigma=10)
        beta2 = pm.Normal('beta2', mu=0, sigma=10)
        sigma = pm.HalfCauchy('sigma', 5)
        mu = pm.Deterministic('mu', alpha + beta1 * [33] + beta2 * np.log([540]))
        y = pm.Normal(name='y', mu=mu, sd=sigma, observed=data['Price'])
        idata = pm.sample(10, tune=10)

    rulari = pm.sample_posterior_predictive(idata, samples=5, model=model1)
    hdi_y = pm.stats.hdi(rulari['y'], hdi_prob=0.90)
    print(hdi_y)
    rulari = pm.sample_posterior_predictive(idata, samples=5, model=model1 , var_names=['mu'])
    hdi_mu = pm.stats.hdi(rulari['mu'], hdi_prob=0.90)
    print(hdi_mu)

if __name__ == "__main__":
    main()