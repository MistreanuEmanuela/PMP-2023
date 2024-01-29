import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import arviz as az

def main():
    # 1a)
    Data = pd.read_csv('BostonHousing.csv')
    x_1 = Data['rm'].values
    x_2 = Data['crim']
    x_3 = Data['indus']
    y = Data['medv'].values
    X = np.column_stack((x_1, x_2, x_3))
    X_mean = X.mean(axis=0, keepdims=True)
    print(X_mean)
    print(y.mean())
    print(X.std(axis=0, keepdims=True))
    print(y.std())

    def scatter_plot(x, y):
        plt.figure(figsize=(15, 5))
        for idx, x_i in enumerate(x.T):
            plt.subplot(1, 4, idx + 1)
        plt.scatter(x_i, y)
        plt.xlabel(f'x_{idx + 1}')
        plt.ylabel(f'y', rotation=0)
        plt.subplot(1, 4, idx + 2)
        plt.scatter(x[:, 0], x[:, 1])
        plt.xlabel(f'x_{idx}')
        plt.ylabel(f'x_{idx + 1}', rotation=0)

    scatter_plot(X,y)
    plt.show()
    # 2b
    with pm.Model() as model:
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=3)
        epsilon = pm.HalfCauchy('epsilon', 5)
        ν = pm.Exponential('ν', 1 / 30)
        X_shared = pm.Data('x_shared', X)
        mu = pm.Deterministic('mu', alpha + pm.math.dot(X_shared, beta))
        y_pred = pm.StudentT('y_pred', mu=mu, sigma=epsilon, nu=ν, observed=y)
        trace = pm.sample(10, tune=10, return_inferencedata=True)
    # simulam 10*2 = 20 extrageri

    # c
    az.plot_forest(trace, hdi_prob=0.95, var_names=['beta'])
    print(az.summary(trace, hdi_prob=0.95, var_names=['beta']))
    plt.show()
    #Observam ca beta 0 are cele mai mari valori asadar variabila de care este atasata aceasta este cea care
    # influenteaza cel mai mult modelul, mai precis "rn"

    # d)
    ppc = pm.sample_posterior_predictive(trace, model=model)
    y_ppc = ppc['y_pred']
    az.plot_posterior({"medv": y_ppc}, hdi_prob=0.5)
    plt.show()
# Observam o medie de aproximativ 21


if __name__ =="__main__":
    main()
