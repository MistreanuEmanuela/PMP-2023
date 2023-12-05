import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

data = pd.read_csv('Admission.csv')
gre = data['GRE'].values
gpa = data['GPA'].values
admission = data['Admission'].values.astype(int)
attributes = ['GRE', 'GPA']
x_1 = data[attributes].values.astype(float)


def main():
    # with pm.Model() as model:
    #     beta0 = pm.Normal('beta0', mu=0, sigma=10)
    #     beta1 = pm.Normal('beta1', mu=0, sigma=10)
    #     beta2 = pm.Normal('beta2', mu=0, sigma=10)
    #     gre_var = pm.Data('gre_var', gre)
    #     gpa_var = pm.Data('gpa_var', gpa)
    #     pi = pm.Deterministic('pi', pm.math.invlogit(beta0 + beta1 * gre_var + beta2 * gpa_var))
    #     admission_likelihood = pm.Bernoulli('admission_likelihood', p=pi, observed=admission)
    #     trace = pm.sample(20, tune=10)

    # sau tinand cont ca avem doar 2 clase:

    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=len(attributes))
        miu = alpha + pm.math.dot(x_1, beta)
        theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-miu)))
        bd = pm.Deterministic('bd', -alpha / beta[0] - beta[1] / beta[0] * x_1[:, 1])
        admission_observed = pm.Bernoulli('admission_observed', p=theta, observed=admission)
        trace = pm.sample(10, tune=10,cores=1, return_inferencedata=True)

    idx = np.argsort(gpa)
    bd = trace.posterior['bd'].mean(('chain', 'draw'))[idx]
    plt.scatter(gpa, gre, c=[f"C{x}" for x in admission])
    plt.plot(gpa[idx], bd, color='k')
    az.plot_hdi(gpa, trace.posterior['bd'], color='k', hdi_prob=0.94)
    plt.xlabel(gpa)
    plt.ylabel(gre)
    plt.show()

    new_data = {'GRE': 550, 'GPA': 3.5}
    GRE = new_data['GRE']
    GPA = new_data['GPA']
    alpha = trace.posterior['alpha'].mean(('chain', 'draw'))
    beta = trace.posterior['beta'].mean(('chain', 'draw'))
    probabilities = np.array(1 / (1 + np.exp(-(alpha + GRE * beta[0] + GPA * beta[1]))))
    lower_bound, upper_bound = az.hdi(probabilities, hdi_prob=0.90)
    hdi_beta1 = (lower_bound, upper_bound)
    print(f"For a student with GRE=550 and GPA=3.5, 90% HDI for probability of admission: {hdi_beta1}")


    new_data = {'GRE': 500, 'GPA': 3.2}
    GRE = new_data['GRE']
    GPA = new_data['GPA']
    alpha = trace.posterior['alpha'].mean(('chain', 'draw'))
    beta = trace.posterior['beta'].mean(('chain', 'draw'))
    probabilities = np.array(1 / (1 + np.exp(-(alpha + GRE * beta[0] + GPA * beta[1]))))
    lower_bound, upper_bound = az.hdi(probabilities, hdi_prob=0.90)
    hdi_beta2 = (lower_bound, upper_bound)
    print(f"For a student with GRE=500 and GPA=3.2, 90% HDI for probability of admission: {hdi_beta2}")


if __name__ == "__main__":
    main()
