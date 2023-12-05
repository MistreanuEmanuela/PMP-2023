import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('Admission.csv')
admission = data['Admission']
gre = data['GRE'].values
gpa = data['GPA'].values

def main():
    with pm.Model() as model:
        beta0 = pm.Normal('beta0', mu=0, sd=10)
        beta1 = pm.Normal('beta1', mu=0, sd=10)
        beta2 = pm.Normal('beta2', mu=0, sd=10)
        pi = pm.Deterministic('pi', pm.math.invlogit(beta0 + beta1 * gre + beta2 * gpa))
        admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed= admission)
        trace = pm.sample(20, tune=10)

    az.plot_trace(trace, var_names=['beta0', 'beta1', 'beta2'])
    plt.show()
    idx = np.argsort(gre)
    bd = trace.posterior['pi'].mean(("chain", "draw"))[idx]
    plt.scatter(gre, gpa, c=[f'C{x}' for x in admission])
    plt.plot(gre[idx], bd, color='k')
    az.plot_hdi(gre, trace.posterior['pi'], color='k')
    plt.xlabel(gre)
    plt.ylabel(gpa)
    plt.show()

if __name__ == "__main__":
    main()
