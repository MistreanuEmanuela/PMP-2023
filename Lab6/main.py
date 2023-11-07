import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt

Y = [0, 5, 10]
theta = [0.2, 0.5]

fig, axes = plt.subplots(len(Y), len(theta), figsize=(12, 8))

for i, Y_val in enumerate(Y):
    for j, theta_val in enumerate(theta):
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)
            name = f'binomial_likelihood_Y{Y_val}_theta{theta_val}'
            likelihood = pm.Binomial(name=name, n=n, p=theta_val, observed=Y_val)

            trace = pm.sample(2000, tune=1000, cores=1)
            posterior_data = az.from_pymc3(trace=trace)

            az.plot_posterior(posterior_data, ax=axes[i, j])
            axes[i, j].set_title(f'Y={Y_val}, θ={theta_val}')

plt.tight_layout()
plt.show()
